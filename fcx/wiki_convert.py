"""Turn a Wikimedia full-history dump into snapshot-sequence samples in the capture Parquet format (non-agent workload, E-WIKI).

Model: a page is a trace and each revision in its window (below) is a snapshot, in dump order; the timestamps are checked to be
nondecreasing within the window. A snapshot's elements are the blank-line-delimited blocks of the revision's main-slot wikitext:
maximal runs of text separated by one or more blank lines (lines of only spaces or tabs count as blank), with surrounding
whitespace stripped and empty runs dropped. An element is the canonical JSON object {"role": "block", "text": <block>} (sorted
keys, no whitespace) and a snapshot is the canonical JSON array of its elements (an empty revision gives "[]"). Revision IDs,
parent IDs and timestamps are used only to validate the input; they are not written: blocks carry no identity but their bytes.

Pages: pass 1 reads the stub dump (metadata, no text). A page is eligible when it is in the main namespace, is not a redirect, has
at least --min-revs revisions, and every revision of its window has main-slot text that is not deleted, the wikitext model, and
a timestamp no earlier than the one before it (a revision without a <model> element is not accepted). The window is all revisions of a page with at most --max-revs of them; otherwise
the --max-revs consecutive revisions starting at xxh3_64("window:<page id>") mod (n - max_revs + 1), which depends on the page
only, not on the seed. For each seed, the --pages eligible pages with the smallest xxh3_64("<seed>:<page id>") form the sample.
Pass 2 streams the full-history dump once, checks that every sampled page appears exactly once with the revision count and the
window timestamps of the stub, and writes every sample. Text inside <content> elements (slots other than the main slot) is
ignored. Every output is written to a staging directory next to the destinations and renamed into place only after all checks
pass; existing destinations are never overwritten.

Output per sample <out>/<name>_s<seed>/: spans.parquet (span_id, trace_id, span_seq), traces.parquet (trace_id, span_lo, span_hi,
n_msgs, repo, exit_status, resolved) and wiki/wide.parquet (span_id, input; one row group per trace): the Parquet schemas of
capture.py. Here n_msgs is the number of blocks of the trace's last snapshot, repo is the title's first character if it is an
ASCII letter (upper-cased) or digit and "other" otherwise, exit_status is "" and resolved is -1. Traces are numbered in dump
order. manifest.parquet lists the sampled pages (seed, trace_id, page_id, title, page revisions, window, first and last
timestamp, sampling hash); wiki_stats.json holds counts, how the snapshots change (see `shape`) and the sampled page IDs
in trace order.
usage: wiki_convert.py --stub <stub-meta-history.xml.gz|bz2|xml> --history <pages-meta-history.xml.bz2 | ... | -> --out <dir>
                       [--name wiki] [--seeds 1,2,3] [--pages 3000] [--min-revs 10] [--max-revs 256]
"""
import argparse, bz2, contextlib, gzip, json, os, re, shutil, sys, time
import xml.etree.ElementTree as ET

import numpy as np, orjson, pyarrow as pa, pyarrow.parquet as pq, xxhash

U32, U64 = pa.uint32(), pa.uint64()
SPLIT = re.compile(r"\n[ \t]*\n")
WIKITEXT = "wikitext"
SPANS = pa.schema([("span_id", U32), ("trace_id", U32), ("span_seq", U32)])
TRACES = pa.schema([("trace_id", U32), ("span_lo", U32), ("span_hi", U32), ("n_msgs", U32), ("repo", pa.string()), ("exit_status", pa.string()),
                    ("resolved", pa.int8())])
WIDE = pa.schema([("span_id", U32), ("input", pa.large_string())])
MANIFEST = pa.schema([("seed", U32), ("trace_id", U32), ("page_id", pa.int64()), ("title", pa.string()), ("page_revisions", U32),
                      ("window_lo", U32), ("window_hi", U32), ("ts_first", pa.string()), ("ts_last", pa.string()), ("sample_hash", U64)])


def opener(path):
    """'-' reads the (already decompressed) dump from standard input, e.g. piped from a parallel bz2 decompressor."""
    if path == "-":
        return contextlib.nullcontext(sys.stdin.buffer)
    return bz2.open(path, "rb") if path.endswith(".bz2") else gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def local(tag):
    return tag.rsplit("}", 1)[-1]


def blocks(text):
    return [p.strip() for p in SPLIT.split(text) if p.strip()]


def request(bl):
    """The canonical serialization of one snapshot's blocks."""
    return b"[" + b",".join(orjson.dumps({"role": "block", "text": p}, option=orjson.OPT_SORT_KEYS) for p in bl) + b"]"


def repo_of(title):
    c = title[:1]
    return c.upper() if len(c) == 1 and c.isascii() and c.isalnum() else "other"


def window(pid, n, w):
    if n <= w:
        return 0, n
    lo = xxhash.xxh3_64_intdigest(f"window:{pid}".encode()) % (n - w + 1)
    return lo, lo + w


def pages(path, keep_text=None):
    """Yield one dict per page in dump order: id, title, ns, redirect, revs. Each revision is (timestamp, model, text state,
    text), where the text state is "ok", "deleted" or "missing" (no main-slot <text>), and text is the main-slot text if
    keep_text maps the page id to a window (lo, hi) that holds the revision's position (else None). Elements are cleared as
    soon as they are read, so memory does not grow with the dump."""
    with opener(path) as f:
        root, page, rev, depth_content = None, None, None, 0
        for ev, el in ET.iterparse(f, events=("start", "end")):
            t = local(el.tag)
            if ev == "start":
                if root is None:
                    root = el
                if t == "page":
                    page = dict(id=None, title=None, ns=None, redirect=False, revs=[])
                elif t == "revision" and page is not None:
                    rev = dict(ts=None, model=None, state="missing", text=None)
                elif t == "content":
                    depth_content += 1
                continue
            if t == "content":
                depth_content -= 1
            elif page is None:
                pass
            elif rev is not None:                                 # inside <revision>
                if depth_content:
                    pass                                          # another slot: ignored
                elif t == "timestamp":
                    rev["ts"] = el.text
                elif t == "model":
                    rev["model"] = el.text
                elif t == "text":
                    if el.get("deleted") is not None:
                        rev["state"] = "deleted"
                    else:
                        rev["state"] = "ok"
                        w = keep_text.get(page["id"]) if keep_text is not None else None
                        if w is not None and w[0] <= len(page["revs"]) < w[1]:
                            rev["text"] = el.text or ""
                elif t == "revision":
                    page["revs"].append((rev["ts"], rev["model"], rev["state"], rev["text"]))
                    rev = None
                    el.clear()
            elif t == "title":
                page["title"] = el.text or ""
            elif t == "ns":
                page["ns"] = int(el.text)
            elif t == "redirect":
                page["redirect"] = True
            elif t == "id" and page["id"] is None:                # outside <revision>: the page id
                page["id"] = int(el.text)
            elif t == "page":
                yield page
                page = None
                root.clear()


def window_problem(revs, lo, hi):
    """Why a window cannot be used, or None."""
    prev = None
    for ts, model, state, _ in revs[lo:hi]:
        if state != "ok":
            return "text_" + state
        if model != WIKITEXT:                                     # a missing <model> is not accepted either
            return "model"
        if ts is None or (prev is not None and ts < prev):
            return "timestamp_order"
        prev = ts
    return None


def shape(snaps):
    """How consecutive snapshots of one trace change, from their block lists."""
    c = dict(snapshots=len(snaps), empty=0, with_duplicate_block=0, identical=0, append_only=0, other_change=0, reentry=0)
    seen, prev = set(), None
    for bl in snaps:
        c["empty"] += not bl
        c["with_duplicate_block"] += len(set(bl)) < len(bl)
        if prev is not None:
            if bl == prev:
                c["identical"] += 1
            elif len(bl) > len(prev) and bl[:len(prev)] == prev:
                c["append_only"] += 1
            else:
                c["other_change"] += 1
            if any(b in seen and b not in prev for b in set(bl)):
                c["reentry"] += 1
            seen.update(prev)
        prev = bl
    return c


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", required=True); ap.add_argument("--history", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="wiki"); ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--pages", type=int, default=3000); ap.add_argument("--min-revs", type=int, default=10)
    ap.add_argument("--max-revs", type=int, default=256)
    a = ap.parse_args(argv)
    try:
        a.seeds = [int(s) for s in a.seeds.split(",")]
    except ValueError:
        ap.error("--seeds: comma-separated integers")
    if len(set(a.seeds)) != len(a.seeds) or min(a.seeds) < 0:
        ap.error("--seeds: distinct nonnegative integers")
    if a.pages < 1 or a.min_revs < 1 or a.max_revs < 1:
        ap.error("--pages, --min-revs and --max-revs must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_]+", a.name):
        ap.error("--name: letters, digits and underscores")
    return a


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    finals = {s: os.path.join(a.out, f"{a.name}_s{s}") for s in a.seeds}
    taken = [d for d in finals.values() if os.path.exists(d)]
    if taken:
        raise SystemExit(f"refusing to overwrite {taken}")

    # pass 1: eligible pages and their windows, from the stub dump
    eligible, ids_seen = {}, set()
    st = dict(pages_in_dump=0, main_namespace_not_redirect=0, with_min_revs=0, excluded={}, eligible=0, capped=0)
    for pg in pages(a.stub):
        pid = pg["id"]
        if pid is None or pid in ids_seen:
            raise SystemExit(f"stub: page id {pid!r} missing or repeated")
        ids_seen.add(pid)
        st["pages_in_dump"] += 1
        if pg["ns"] != 0 or pg["redirect"]:
            continue
        st["main_namespace_not_redirect"] += 1
        n = len(pg["revs"])
        if n < a.min_revs:
            continue
        st["with_min_revs"] += 1
        lo, hi = window(pid, n, a.max_revs)
        why = window_problem(pg["revs"], lo, hi)
        if why:
            st["excluded"][why] = st["excluded"].get(why, 0) + 1
            continue
        eligible[pid] = (n, lo, hi, pg["revs"][lo][0], pg["revs"][hi - 1][0], pg["title"])
        st["capped"] += n > a.max_revs
    st["eligible"] = len(eligible)
    del ids_seen
    print(f"pass 1: {json.dumps(st)} {time.time() - t0:.0f}s", flush=True)
    if len(eligible) < a.pages:
        raise SystemExit(f"only {len(eligible)} eligible pages, fewer than --pages {a.pages}")
    ids = np.array(sorted(eligible), dtype=np.int64)
    sample, hashes = {}, {}
    for s in a.seeds:
        h = np.array([xxhash.xxh3_64_intdigest(f"{s}:{p}".encode()) for p in ids], dtype=np.uint64)
        pick = np.argsort(h, kind="stable")[:a.pages]
        sample[s] = set(ids[pick].tolist())
        hashes[s] = dict(zip(ids[pick].tolist(), h[pick].tolist()))
    wanted = set().union(*sample.values())

    # pass 2: the samples from the full-history dump, into a staging directory
    os.makedirs(a.out, exist_ok=True)
    stage = os.path.join(a.out, f".{a.name}.staging-{os.getpid()}")
    os.makedirs(stage)
    done, writers, renamed = False, {}, []
    try:
        z = dict(compression="zstd", compression_level=3)
        x = {}
        for s in a.seeds:
            d = os.path.join(stage, f"{a.name}_s{s}")
            os.makedirs(os.path.join(d, "wiki"))
            writers[s] = pq.ParquetWriter(os.path.join(d, "wiki", "wide.parquet"), WIDE, **z)
            x[s] = dict(dir=d, spans=([], [], []), traces=[[] for _ in TRACES], manifest=[[] for _ in MANIFEST], n_traces=0, n_spans=0,
                        logical_bytes=0, blocks=0, shape={}, seen=set())
        n_hist = 0
        for pg in pages(a.history, keep_text={p: eligible[p][1:3] for p in wanted}):
            n_hist += 1
            pid = pg["id"]
            if pid not in wanted:
                continue
            n, lo, hi, ts_first, ts_last, title = eligible[pid]
            revs = pg["revs"]
            if len(revs) != n:
                raise SystemExit(f"page {pid}: {len(revs)} revisions in the history dump, {n} in the stub")
            if window_problem(revs, lo, hi) or (revs[lo][0], revs[hi - 1][0]) != (ts_first, ts_last) or pg["title"] != title:
                raise SystemExit(f"page {pid}: window differs from the stub")
            snaps = [blocks(r[3]) for r in revs[lo:hi]]
            reqs = [request(bl) for bl in snaps]
            texts = [b.decode() for b in reqs]
            sh = shape(snaps)
            for s in a.seeds:
                if pid not in sample[s]:
                    continue
                y = x[s]
                if pid in y["seen"]:
                    raise SystemExit(f"page {pid} appears twice in the history dump")
                y["seen"].add(pid)
                k, base = y["n_traces"], y["n_spans"]
                span_ids = np.arange(base, base + len(texts), dtype=np.uint32)
                writers[s].write_table(pa.table({"span_id": pa.array(span_ids, U32), "input": pa.array(texts, pa.large_string())}, schema=WIDE))
                y["spans"][0].append(span_ids); y["spans"][1].append(np.full(len(texts), k, np.uint32)); y["spans"][2].append(np.arange(len(texts), dtype=np.uint32))
                for col, v in zip(y["traces"], (k, base, base + len(texts) - 1, len(snaps[-1]), repo_of(title), "", -1)):
                    col.append(v)
                for col, v in zip(y["manifest"], (s, k, pid, title, n, lo, hi, ts_first, ts_last, hashes[s][pid])):
                    col.append(v)
                y["n_traces"] += 1; y["n_spans"] += len(texts); y["logical_bytes"] += sum(len(b) for b in reqs)
                y["blocks"] += sum(len(bl) for bl in snaps)
                for key, v in sh.items():
                    y["shape"][key] = y["shape"].get(key, 0) + v
        for s in a.seeds:
            writers[s].close()
            missing = sample[s] - x[s]["seen"]
            if missing:
                raise SystemExit(f"seed {s}: {len(missing)} sampled pages not found in the history dump")
        if n_hist != st["pages_in_dump"]:
            raise SystemExit(f"{n_hist} pages in the history dump, {st['pages_in_dump']} in the stub")
        for s in a.seeds:
            y, d = x[s], x[s]["dir"]
            pq.write_table(pa.table({"span_id": pa.array(np.concatenate(y["spans"][0]), U32), "trace_id": pa.array(np.concatenate(y["spans"][1]), U32),
                                     "span_seq": pa.array(np.concatenate(y["spans"][2]), U32)}, schema=SPANS), os.path.join(d, "spans.parquet"), **z)
            pq.write_table(pa.table(dict(zip(TRACES.names, y["traces"])), schema=TRACES), os.path.join(d, "traces.parquet"), **z)
            pq.write_table(pa.table(dict(zip(MANIFEST.names, y["manifest"])), schema=MANIFEST), os.path.join(d, "manifest.parquet"), **z)
            out = dict(src=dict(stub=os.path.basename(a.stub), history=os.path.basename(a.history) if a.history != "-" else "stdin"), seed=s,
                       pages=a.pages, min_revs=a.min_revs, max_revs=a.max_revs, pass1=st, traces=y["n_traces"], spans=y["n_spans"],
                       logical_bytes=y["logical_bytes"], blocks=y["blocks"], capped_in_sample=int(sum(eligible[p][0] > a.max_revs for p in sample[s])),
                       shape=y["shape"], n_msgs="number of blocks of the trace's last snapshot", page_ids=y["manifest"][2],
                       seconds=round(time.time() - t0, 1))
            with open(os.path.join(d, "wiki_stats.json"), "w") as f:
                json.dump(out, f, indent=1)
            print("sample", s, json.dumps(out), flush=True)
        taken = [d for d in finals.values() if os.path.exists(d)]
        if taken:
            raise SystemExit(f"refusing to overwrite {taken}")
        for s in a.seeds:
            os.rename(x[s]["dir"], finals[s])
            renamed.append(finals[s])
        done = True
    finally:
        for w in writers.values():
            with contextlib.suppress(Exception):
                w.close()
        if not done:
            for d in renamed:                                     # destinations this run created before it failed
                shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
    print("WIKI_CONVERT_OK", flush=True)


if __name__ == "__main__":
    main()
