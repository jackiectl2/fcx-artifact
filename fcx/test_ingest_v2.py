"""E-FAST differential tests: the sped-up fcx/ingest.py against the frozen reference fcx/ingest_v1.py (production contract:
non-negative integer message IDs from the dictionary, canonical JSON-object messages, overflow is a terminal error).
  1. Differencer, exhaustively over all pairs of contexts of length 0-4 over 3 IDs, and on random edit sequences (appends,
     drops, shortened requests, replacements, insertions, duplicates, empty contexts, rewrites, prefix or suffix covering almost
     all of the request) for budgets 0, 2, 64 and GAP = 2^32, 4, 2 (forced relabelling, also after new occurrences were
     already numbered in the same call): identical transitions, checkpoints (values, shapes, dtypes), next_occ, fallbacks and
     state; the caller's list may be changed after the call without effect.
  2. Order-key overflow: both versions raise OverflowError at the same request, on the append path (GAP = 2^63) and on the
     relabel path (adjacent keys made with GAP = 1, then GAP = 2^63 and three insertions between them; entering _relabel is
     asserted).
  3. recurring() against the reference formula (the whole previous messages inside the maximal equal byte run, numpy
     comparison + searchsorted) at every message boundary of random request pairs.
  4. Ingester.request_text and Ingester.request of both versions on random interleaved request streams with near-copies (a
     random unfinished trace at each step, call order kept per trace, an A-B-A interleaving asserted): identical return
     values and fallback counts.
usage: python test_ingest_v2.py   (prints TEST_INGEST_V2_OK)"""
import itertools, os, random

import numpy as np, orjson

import ingest, ingest_v1

TMP = os.environ.get("TMPDIR", "/tmp")


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def set_gap(g):
    ingest.GAP = ingest_v1.GAP = g


def state_equal(d1, d2):
    return (d1.occ.tolist() == d2.occ and d1.ord.tolist() == d2.ord and d1.ids.tolist() == d2.ids and d1.next_occ == d2.next_occ
            and d1.fallbacks == d2.fallbacks and d1.calls == d2.calls)


def feed_both(d1, d2, ctx):
    """Feed ctx to both; v2 gets a list that is changed afterwards. Returns (equal, exception names)."""
    e1 = e2 = None
    try:
        r1 = d1.feed(list(ctx))
    except OverflowError:
        e1 = "OverflowError"
    inp = list(ctx)
    try:
        r2 = d2.feed(inp)
    except OverflowError:
        e2 = "OverflowError"
    inp.append(10 ** 6); inp[:1] = [10 ** 6 + 1]                  # the caller reuses its list
    if e1 or e2:
        return e1 == e2, e1
    return same(r1, r2) and state_equal(d1, d2), None


def exhaustive(budget, gap):
    set_gap(gap)
    ctxs = [list(c) for L in range(5) for c in itertools.product(range(3), repeat=L)]
    bad = 0
    for a, b in itertools.product(ctxs, ctxs):
        d1, d2 = ingest_v1.Differencer(budget, 2), ingest.Differencer(budget, 2)
        for c in (a, b, a):
            ok, _ = feed_both(d1, d2, c)
            bad += not ok
    return bad, len(ctxs) ** 2


def random_edits(budget, gap, n_traces=400, seed=0):
    set_gap(gap)
    rng, bad, relabels, skipped = random.Random(seed), 0, 0, 0
    for _ in range(n_traces):
        d1, d2 = ingest_v1.Differencer(budget, 3), ingest.Differencer(budget, 3)
        ctx, fresh = [], 100
        for _ in range(rng.randint(1, 50)):
            r = rng.random()
            if r < 0.04:
                ctx = []
            elif r < 0.08:
                ctx = [rng.randint(0, 9) for _ in range(rng.randint(0, 25))]
            elif r < 0.14 and ctx:
                ctx = ctx[:rng.randint(0, len(ctx))]               # shortened request
            elif r < 0.20 and len(ctx) > 2:                       # one change inside a long shared prefix and suffix
                i = rng.randint(1, len(ctx) - 2); ctx[i] = rng.randint(0, 9)
            else:
                if rng.random() < 0.3 and ctx:
                    i = rng.randint(0, len(ctx) - 1); del ctx[i:i + rng.randint(1, 4)]
                if rng.random() < 0.3 and ctx:
                    ctx[rng.randint(0, len(ctx) - 1)] = rng.randint(0, 9)
                for _ in range(rng.choice([0, 0, 1, 3])):         # middle insertions (several runs: relabel after numbering)
                    ctx.insert(rng.randint(0, len(ctx)), rng.choice([rng.randint(0, 9), fresh := fresh + 1]))
                for _ in range(rng.randint(0, 3)):
                    fresh += 1; ctx.append(rng.choice([fresh, rng.randint(0, 9)]))
            before = (d1.next_occ, d1.fallbacks)
            ok, _ = feed_both(d1, d2, ctx)
            bad += not ok
            if d1.fallbacks > before[1]:
                relabels += 1
                opened = len(set(d1.occ.tolist()) - set(range(before[0])))
                skipped += d1.next_occ - before[0] > opened        # occurrence numbers consumed before a relabel
    return bad, relabels, skipped


class Relabels(ingest.Differencer):                               # v2 that records entering the relabel path
    entered = 0

    def _relabel(self, ids, p):
        Relabels.entered += 1
        return super()._relabel(ids, p)


def overflow():
    """Append path: GAP = 2^63 overflows on the second appended key. Relabel path: keys 1 and 2 are made adjacent with GAP = 1,
    then GAP = 2^63 and three messages between them: no room, relabel, and the relabelled keys overflow."""
    out = {}
    for name, steps_spec in (("append", [(1 << 63, [1]), (1 << 63, [1, 2]), (1 << 63, [1, 2, 3])]),
                             ("relabel", [(1, [1, 2]), (1 << 63, [1, 3, 4, 5, 2])])):
        d1, d2 = ingest_v1.Differencer(64, 1), Relabels(64, 1)
        Relabels.entered, steps = 0, []
        for gap, c in steps_spec:
            set_gap(gap)
            ok, exc = feed_both(d1, d2, c)
            steps.append((ok, exc))
            if exc:
                break
        out[name] = dict(steps=steps, relabel_entered=Relabels.entered)
        assert all(ok for ok, _ in steps) and steps[-1][1] == "OverflowError", (name, steps)
        assert (Relabels.entered > 0) == (name == "relabel"), (name, Relabels.entered)
    set_gap(1 << 32)
    return out


def reference_recurring(text, pos, ptext, pends, k):
    x, y = np.frombuffer(text, np.uint8), np.frombuffer(ptext, np.uint8)
    ps = int(pends[k - 1]) + 1 if k else 1
    lim = min(len(text) - pos, len(ptext) - ps)
    d = np.flatnonzero(x[pos:pos + lim] != y[ps:ps + lim])
    run = int(d[0]) if d.size else lim
    return int(np.searchsorted(pends[k:], ps + run, side="right"))


def recurring_oracle(trials=3000, seed=1):
    rng, bad, cases = random.Random(seed), 0, 0
    dump = lambda m: orjson.dumps(m, option=orjson.OPT_SORT_KEYS)
    pool = [dump({"content": "x" * rng.randint(0, 6) + rng.choice("ab"), "role": rng.choice(["user", "tool"])}) for _ in range(12)]
    for _ in range(trials):
        prev = [rng.choice(pool) for _ in range(rng.randint(1, 8))]
        cur = list(prev)
        for _ in range(rng.randint(0, 3)):
            r = rng.random()
            if r < 0.3 and cur:
                del cur[rng.randint(0, len(cur) - 1)]
            elif r < 0.6:
                cur.insert(rng.randint(0, len(cur)), rng.choice(pool))
            elif cur:
                i = rng.randint(0, len(cur) - 1); cur[i] = cur[i][:-2] + b"z}"     # same length, last bytes differ
        if not cur:
            continue
        ptext, text = b"[" + b",".join(prev) + b"]", b"[" + b",".join(cur) + b"]"
        pends, o = [], 1
        for m in prev:
            o += len(m); pends.append(o); o += 1
        starts, o = [], 1
        for m in cur:
            starts.append(o); o += len(m) + 1
        for k in range(len(prev)):
            for pos in starts:
                cases += 1
                bad += ingest.recurring(text, pos, memoryview(ptext), pends, k, len(prev)) != reference_recurring(text, pos, ptext, np.array(pends, np.int64), k)
    return bad, cases


def streams(n_traces=120, seed=2):
    """Both Ingesters, request_text and request, on interleaved random streams with near-copies and repeats."""
    rng, dump, fresh = random.Random(seed), (lambda m: orjson.dumps(m, option=orjson.OPT_SORT_KEYS)), [0]

    def message(like=None):
        if like is not None and rng.random() < 0.7:
            m = orjson.loads(like)
            m["content"] = m["content"][:-1] + rng.choice("xyz") if rng.random() < 0.5 else m["content"] + " more"
            return dump(m)
        fresh[0] += 1
        return dump({"content": "head " * rng.choice([0, 1, 8]) + "body %d " % fresh[0] + "pad " * rng.randint(0, 300), "role": rng.choice(["user", "assistant", "tool"])})

    reqs = []
    for t in range(n_traces):
        ctx = [message(), message()]
        for _ in range(rng.randint(1, 25)):
            r = rng.random()
            if r < 0.05:
                ctx = [message() for _ in range(rng.randint(0, 6))]
            elif r > 0.1:
                if rng.random() < 0.3 and len(ctx) > 2:
                    i = rng.choice([1, rng.randint(1, len(ctx) - 1)]); del ctx[i:i + rng.randint(1, 3)]
                if rng.random() < 0.4 and ctx:
                    i = rng.randint(0, len(ctx) - 1); ctx[i] = message(ctx[i])
                if rng.random() < 0.2 and ctx:
                    ctx.insert(rng.randint(0, len(ctx)), message(rng.choice(ctx)))
                if rng.random() < 0.1 and ctx:
                    ctx.append(rng.choice(ctx))
                ctx += [message() for _ in range(rng.randint(0, 3))]
            reqs.append((t, list(ctx)))
    by_trace = {}
    for i, (t, _) in enumerate(reqs):
        by_trace.setdefault(t, []).append(i)
    order, live, pos = [], sorted(by_trace), {t: 0 for t in by_trace}
    while live:                                                   # a random live trace at each step; call order kept per trace
        t = rng.choice(live)
        order.append(by_trace[t][pos[t]]); pos[t] += 1
        if pos[t] == len(by_trace[t]):
            live.remove(t)
    tr = [reqs[i][0] for i in order]
    assert any(a == c != b for a, b, c in zip(tr, tr[1:], tr[2:])), "the stream is not interleaved"
    bad = 0
    for mode in ("text", "list"):
        ings = [mod.Ingester(mod.Dictionary(os.path.join(TMP, f"v2test_{mod.__name__}_{mode}.bin")), 64, 4) for mod in (ingest_v1, ingest)]
        for i in order:
            t, ctx = reqs[i]
            text = b"[" + b",".join(ctx) + b"]"
            res = [ing.request_text(t, text) if mode == "text" else ing.request(t, list(ctx)) for ing in ings]
            bad += not same(*res)
        for t in {t for t, _ in reqs}:
            bad += ings[0].active[t].fallbacks != ings[1].active[t].fallbacks
            for ing in ings:
                ing.end(t)
        for ing, mod in zip(ings, (ingest_v1, ingest)):
            os.close(ing.dict.fd); os.remove(os.path.join(TMP, f"v2test_{mod.__name__}_{mode}.bin"))
    return bad, len(reqs)


if __name__ == "__main__":
    report = {}
    for budget in (0, 2, 64):
        for gap in (1 << 32, 4, 2):
            report[f"exhaustive b{budget} gap{gap}"] = exhaustive(budget, gap)
            assert report[f"exhaustive b{budget} gap{gap}"][0] == 0, report
            report[f"random b{budget} gap{gap}"] = random_edits(budget, gap, seed=budget * 7 + gap % 97)
            assert report[f"random b{budget} gap{gap}"][0] == 0, report
    assert any(v[1] > 0 and v[2] > 0 for k, v in report.items() if k.startswith("random") and ("gap2" in k or "gap4" in k)), "no relabel after numbering"
    report["overflow"] = overflow()
    set_gap(1 << 32)
    report["recurring"] = recurring_oracle()
    assert report["recurring"][0] == 0 and report["recurring"][1] > 10000, report["recurring"]
    report["streams"] = streams()
    assert report["streams"][0] == 0, report["streams"]
    for k, v in report.items():
        print(k, v)
    print("TEST_INGEST_V2_OK")
