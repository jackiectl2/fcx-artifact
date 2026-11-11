"""Capture the requests an agent framework sends, by driving the framework's own context-management code over recorded trajectories.

A recorded trajectory gives the messages of an agent run in order. For every LLM call (one per assistant message) the context
manager under test receives the history up to that call and returns the messages it would send; that list, serialized as a
JSON array of provider-format messages (canonical: sorted keys, no whitespace), is the request. No model is called: the
recorded assistant messages are the model's replies. The managers are the installed framework code, not reimplementations:

  prefix            no context management: every earlier message
  swea_last5        SWE-agent LastNObservations(n=5)                       (the classic SWE-agent configuration)
  swea_cache2       SWE-agent CacheControlHistoryProcessor(last_n_messages=2)   (SWE-agent's default for Anthropic models)
  swea_window       SWE-agent ClosedWindowHistoryProcessor
  autogen_buf32     AutoGen BufferedChatCompletionContext(buffer_size=32)
  autogen_ht        AutoGen HeadAndTailChatCompletionContext(head_size=2, tail_size=30)
  lc_trim32k        LangChain trim_messages(max_tokens=32000, token_counter="approximate", strategy="last", include_system=True, start_on=("human", "ai"))
  litellm_trim32k   LiteLLM trim_messages(max_tokens=32000)
  litellm_cache     LiteLLM AnthropicCacheControlHook with injection points on the system message and on the last message
                    (prompt-cache breakpoints for a Claude model; the mark on the last message moves at every call)

Output: <out>/<manager>/wide.parquet with one row per LLM call (span_id, input), one row group per trajectory;
<out>/spans.parquet, <out>/traces.parquet, <out>/capture_stats.json. Span IDs are dense in (trace, call order).
"""
import argparse, asyncio, copy, glob, json, os, time, types

import numpy as np, orjson, pyarrow as pa, pyarrow.parquet as pq

U32 = pa.uint32()
dumps = lambda m: orjson.dumps(m, option=orjson.OPT_SORT_KEYS)


# ---- corpora: one trajectory -> (history, tools). `history` holds provider-format messages (OpenAI chat format), or SWE-agent
# history items when the corpus was recorded by SWE-agent.

def clean(m):
    return {k: v for k, v in m.items() if v is not None}


def load_openhands(row, col):                                     # SWE-rebench OpenHands, SWE-Gym OpenHands: chat messages with tool calls
    return [clean(m) for m in row[col]], row.get("tools")


def load_smith(row, col):                                         # SWE-smith: the history items SWE-agent saved
    hist = orjson.loads(row[col])
    for h in hist:                                                # marks left by the last request of the run
        h.pop("cache_control", None)
        if isinstance(h["content"], list):
            for block in h["content"]:
                block.pop("cache_control", None)
    return hist, None


def load_sweagent(row, col):                                      # SWE-agent 0.x trajectories: system prompt, observations as user messages
    hist = []
    for m in row[col]:
        if m["role"] == "system":
            hist.append(dict(role="system", content=m["system_prompt"] or m["text"] or "", agent="main", message_type="system_prompt"))
        elif m["role"] == "ai":
            hist.append(dict(role="assistant", content=m["text"] or "", agent="main", message_type="action"))
        else:
            hist.append(dict(role="user", content=m["text"] or "", agent="main", message_type="observation"))
    return hist, None


FORMATS = {"openhands": (load_openhands, "trajectory", False), "swegym": (load_openhands, "messages", False),
           "smith": (load_smith, "messages", True), "sweagent": (load_sweagent, "trajectory", True)}     # loader, column, SWE-agent history?


# ---- SWE-agent

def swea_messages(history):
    """SWE-agent's own conversion of (processed) history items into the messages of the API call."""
    from sweagent.agent.models import LiteLLMModel
    stub = types.SimpleNamespace(config=types.SimpleNamespace(convert_system_to_user=False), logger=types.SimpleNamespace(debug=lambda *a, **k: None))
    return LiteLLMModel._history_to_messages(stub, history)


def m_swea(make):
    def run(tr):
        procs = make()
        for k in tr.calls:
            h = tr.history[:k]                                    # the same item objects on every call: processors edit them in place, as in the agent
            for proc in procs:
                h = proc(h)
            yield swea_messages(h)
    return run


def swea(name):
    def make():
        import sweagent.agent.history_processors as hp
        return {"swea_last5": lambda: [hp.LastNObservations(n=5)], "swea_cache2": lambda: [hp.CacheControlHistoryProcessor(last_n_messages=2)],
                "swea_window": lambda: [hp.ClosedWindowHistoryProcessor()]}[name]()
    return m_swea(make)


# ---- AutoGen

def text_of(m):
    c = m.get("content")
    return c if isinstance(c, str) else "" if c is None else "".join(b.get("text", "") for b in c)


def m_autogen(name):
    def run(tr):
        from autogen_core import FunctionCall
        from autogen_core.model_context import BufferedChatCompletionContext, HeadAndTailChatCompletionContext
        from autogen_core.models import AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage, UserMessage
        ctx = BufferedChatCompletionContext(buffer_size=32) if name == "autogen_buf32" else HeadAndTailChatCompletionContext(head_size=2, tail_size=30)
        loop, back, nxt = tr.loop, {}, 0
        head = [m for m in tr.msgs if m["role"] == "system"]      # an AutoGen agent sends its system messages ahead of the context
        for k in tr.calls:
            for m in tr.msgs[nxt:k]:
                if m["role"] == "system":
                    continue
                if m["role"] == "assistant" and m.get("tool_calls"):
                    obj = AssistantMessage(content=[FunctionCall(id=c["id"], name=c["function"]["name"], arguments=c["function"]["arguments"]) for c in m["tool_calls"]],
                                           thought=text_of(m) or None, source="assistant")
                elif m["role"] == "assistant":
                    obj = AssistantMessage(content=text_of(m), source="assistant")
                elif m["role"] == "tool":
                    obj = FunctionExecutionResultMessage(content=[FunctionExecutionResult(content=text_of(m), name=m.get("name") or "tool", call_id=m["tool_call_id"], is_error=False)])
                else:
                    obj = UserMessage(content=text_of(m), source="user")
                back[id(obj)] = m
                loop.run_until_complete(ctx.add_message(obj))
            nxt = k
            yield head + [back[id(o)] if id(o) in back else {"role": "user", "content": o.content} for o in loop.run_until_complete(ctx.get_messages())]
    return run


# ---- LangChain, LiteLLM

def m_langchain(tr):
    from langchain_core.messages import convert_to_messages, trim_messages
    from langchain_core.messages.utils import convert_to_openai_messages
    lc = []
    for m in tr.msgs:
        d = dict(m, content=m.get("content") or "")
        try:
            lc += convert_to_messages([d])
        except json.JSONDecodeError:                              # tool-call arguments that are not JSON (the model's mistake, recorded as such)
            good, bad = [], []                                    # LangChain keeps such a call as an invalid tool call with its raw arguments
            for c in d.pop("tool_calls"):
                try:
                    json.loads(c["function"]["arguments"], strict=False); good.append(c)
                except json.JSONDecodeError:
                    bad.append(dict(name=c["function"]["name"], args=c["function"]["arguments"], id=c["id"], error=None, type="invalid_tool_call"))
            msg = convert_to_messages([dict(d, tool_calls=good)])[0]
            msg.invalid_tool_calls = bad
            lc.append(msg)
    for k in tr.calls:
        kept = trim_messages(lc[:k], max_tokens=32000, token_counter="approximate", strategy="last", include_system=True, start_on=("human", "ai"))
        out = convert_to_openai_messages(kept)
        for o, msg in zip(out, kept):                             # as langchain-openai does, an invalid tool call goes back with its raw arguments
            for b in getattr(msg, "invalid_tool_calls", None) or []:
                o.setdefault("tool_calls", []).append({"type": "function", "id": b["id"], "function": {"name": b["name"], "arguments": b["args"]}})
        yield out


def m_litellm(tr):
    from litellm.utils import trim_messages
    for k in tr.calls:
        yield trim_messages(tr.msgs[:k], max_tokens=32000)


def m_litellm_cache(tr):
    from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
    hook = AnthropicCacheControlHook()
    for k in tr.calls:
        points = [{"location": "message", "role": "system"}, {"location": "message", "index": -1}]
        yield hook.get_chat_completion_prompt(model="claude-sonnet-4-5", messages=tr.msgs[:k], non_default_params={"cache_control_injection_points": points},
                                              prompt_id=None, prompt_variables=None, dynamic_callback_params={})[1]


MANAGERS = {"prefix": lambda tr: (tr.msgs[:k] for k in tr.calls), "litellm_cache": m_litellm_cache, "swea_last5": swea("swea_last5"), "swea_cache2": swea("swea_cache2"), "swea_window": swea("swea_window"),
            "autogen_buf32": m_autogen("autogen_buf32"), "autogen_ht": m_autogen("autogen_ht"), "lc_trim32k": m_langchain, "litellm_trim32k": m_litellm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Parquet file, or a glob of files, with one trajectory per row")
    ap.add_argument("--format", required=True, choices=FORMATS); ap.add_argument("--out", required=True)
    ap.add_argument("--managers", required=True); ap.add_argument("--sample", type=int, default=0, help="number of trajectories drawn at random (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    managers = a.managers.replace(":", ",").split(",")
    load, col, is_swea = FORMATS[a.format]
    files = sorted(glob.glob(a.src))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    keep = set(np.random.default_rng(a.seed).choice(total, a.sample, replace=False).tolist()) if a.sample else None
    names = pq.ParquetFile(files[0]).schema_arrow.names
    meta = [c for c in ("repo", "instance_id", "exit_status", "resolved", "target", "tools") if c in names]
    for m in managers:
        os.makedirs(f"{a.out}/{m}", exist_ok=True)
    z = dict(compression="zstd", compression_level=3)
    wide = {m: pq.ParquetWriter(f"{a.out}/{m}/wide.parquet", pa.schema([("span_id", U32), ("input", pa.large_string())]), **z) for m in managers}
    spans, traces = ([], [], []), ([], [], [], [], [], [], [])
    stats = {m: dict(requests=0, logical_bytes=0, messages_sent=0, seconds=0.0) for m in managers}
    loop = asyncio.new_event_loop()
    row_no = n_traces = n_spans = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=[col] + meta):
            idx = [i for i in range(batch.num_rows) if keep is None or row_no + i in keep]
            row_no += batch.num_rows
            for row in batch.take(pa.array(idx, pa.int64())).to_pylist() if idx else []:
                history, tools = load(row, col)
                roles = [h["role"] for h in history]
                calls = [k for k, r in enumerate(roles) if r == "assistant"]
                if not calls:
                    continue
                span_ids = np.arange(n_spans, n_spans + len(calls), dtype=np.uint32)
                head = [{"role": "tools", "content": tools}] if tools else []     # tool definitions are sent with every call
                for m in managers:
                    t0 = time.perf_counter()
                    tr = types.SimpleNamespace(calls=calls, loop=loop, history=copy.deepcopy(history) if is_swea else None)
                    tr.msgs = swea_messages(history) if is_swea else history
                    reqs = [head + list(req) for req in MANAGERS[m](tr)]
                    assert len(reqs) == len(calls), (m, len(reqs), len(calls))
                    texts = [b"[" + b",".join(map(dumps, req)) + b"]" for req in reqs]
                    st = stats[m]
                    st["requests"] += len(texts); st["logical_bytes"] += sum(map(len, texts)); st["messages_sent"] += sum(map(len, reqs))
                    wide[m].write_table(pa.table({"span_id": pa.array(span_ids, U32), "input": pa.array([t.decode() for t in texts], pa.large_string())}))
                    st["seconds"] += time.perf_counter() - t0
                repo = row.get("repo") or (row["instance_id"].rsplit("-", 1)[0] if row.get("instance_id") else None)
                res = row.get("resolved", row.get("target"))
                for buf, v in zip(traces, (n_traces, n_spans, n_spans + len(calls) - 1, len(history), repo, row.get("exit_status"), -1 if res is None else int(res))):
                    buf.append(v)
                spans[0].append(span_ids); spans[1].append(np.full(len(calls), n_traces, np.uint32)); spans[2].append(np.arange(len(calls), dtype=np.uint32))
                n_traces += 1; n_spans += len(calls)
                if n_traces % 500 == 0:
                    print(n_traces, "trajectories", n_spans, "requests", {m: round(s["seconds"]) for m, s in stats.items()}, flush=True)
    for w in wide.values():
        w.close()
    pq.write_table(pa.table({"span_id": pa.array(np.concatenate(spans[0]), U32), "trace_id": pa.array(np.concatenate(spans[1]), U32),
                             "span_seq": pa.array(np.concatenate(spans[2]), U32)}), f"{a.out}/spans.parquet", **z)
    pq.write_table(pa.table({"trace_id": pa.array(traces[0], U32), "span_lo": pa.array(traces[1], U32), "span_hi": pa.array(traces[2], U32), "n_msgs": pa.array(traces[3], U32),
                             "repo": pa.array(traces[4], pa.string()), "exit_status": pa.array(traces[5], pa.string()), "resolved": pa.array(traces[6], pa.int8())}),
                   f"{a.out}/traces.parquet", **z)
    out = dict(src=a.src, format=a.format, corpus_trajectories=total, sample=a.sample, seed=a.seed, traces=n_traces, spans=n_spans, managers=stats)
    json.dump(out, open(f"{a.out}/capture_stats.json", "w"), indent=1)
    print(json.dumps(out, indent=1)); print("CAPTURE_OK")


if __name__ == "__main__":
    main()
