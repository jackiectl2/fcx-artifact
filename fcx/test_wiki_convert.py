"""Test of wiki_convert.py on synthetic MediaWiki 0.11 dumps.

Checks: block splitting and canonical requests; the eligibility rule (main namespace, not a redirect, at least --min-revs
revisions, a window with main-slot wikitext that is not deleted and nondecreasing timestamps), with a deleted revision outside
the window not excluding a page; hash-chosen windows of long pages; hash sampling with more eligible pages than --pages; other
slots ignored; empty snapshots ("[]"); exact Arrow schemas; the manifest and the shape counts; file and standard input giving
identical output; the output through the ingestion path (delta parsing equal to full parsing, replay reproducing every request,
one array row per span) and through ingest_ablate.py (every exact variant reproduces every request); and the failures: a
sampled page missing from the history, a revision-count or window mismatch between stub and history, a repeated page id,
too few eligible pages, invalid arguments, an existing destination (left untouched), and a clean rerun after a failure.
"""
import bz2, gzip, json, os, re, subprocess, sys, tempfile
from xml.sax.saxutils import escape

import orjson, pyarrow as pa, pyarrow.parquet as pq, xxhash

from ingest import Dictionary, Ingester
from ingest_ablate import KeyReplay
from wiki_convert import MANIFEST, SPANS, TRACES, WIDE, blocks, repo_of, request, shape

HERE = os.path.dirname(os.path.abspath(__file__))
NS = "http://www.mediawiki.org/xml/export-0.11/"
W, MIN = 6, 5                                                    # --max-revs and --min-revs of the test


def rev_xml(i, r, with_text):
    ts, text = r["ts"], r["text"]
    model = r.get("model", "wikitext")
    model_xml = "" if r.get("nomodel") else f"<model>{model}</model>"
    if r.get("deleted"):
        t = '<text deleted="deleted" />'
    elif with_text:
        t = f'<text bytes="{len(text.encode())}" xml:space="preserve">{escape(text)}</text>'
    else:
        t = f'<text bytes="{len(text.encode())}" id="{i}" />'
    slot = ""
    if r.get("slot"):                                            # another slot: must be ignored
        slot = f"<content><role>mediainfo</role><origin>{i}</origin><model>json</model><format>application/json</format>" + \
               (f'<text bytes="9" xml:space="preserve">{escape(r["slot"])}</text>' if with_text else '<text bytes="9" id="1" />') + "</content>"
    return (f"<revision><id>{i}</id><parentid>{i - 1}</parentid><timestamp>{ts}</timestamp><contributor><username>u</username><id>7</id></contributor>"
            f"{model_xml}<format>text/x-wiki</format>{t}{slot}<sha1>x</sha1></revision>")


def dump(pages, with_text):
    head = f'<mediawiki xmlns="{NS}" version="0.11" xml:lang="en"><siteinfo><sitename>t</sitename><namespaces><namespace key="0" /></namespaces></siteinfo>'
    body = "".join(f"<page><title>{escape(p['title'])}</title><ns>{p['ns']}</ns><id>{p['id']}</id>" + ('<redirect title="X" />' if p.get("redirect") else "")
                   + "".join(rev_xml(p["id"] * 1000 + i, r, with_text) for i, r in enumerate(p["revs"])) + "</page>" for p in pages)
    return (head + body + "</mediawiki>").encode()


def ts(k):
    return f"2020-01-{1 + k // 1440:02d}T{(k // 60) % 24:02d}:{k % 60:02d}:00Z"


def texts_for(pid, n):
    """A revision history with appends, mid insertions, deletions, a reorder, a duplicate block and an empty revision."""
    out, cur = [], [f"Intro of {pid}.", f"Body {pid} one."]
    for k in range(n):
        op = (pid + k) % 6
        if op == 0:
            cur = cur + [f"Tail {pid}-{k}."]
        elif op == 1 and len(cur) > 1:
            cur = cur[:1] + [f"Mid {pid}-{k} = (x) [[Link]] {{{{cite web}}}} \"q\" & <ref>é 中</ref>"] + cur[1:]
        elif op == 2 and len(cur) > 2:
            cur = cur[:1] + cur[2:]
        elif op == 3 and len(cur) > 1:
            cur = [cur[1], cur[0]] + cur[2:]
        elif op == 4:
            cur = cur + [cur[0]]
        out.append("" if (pid % 2 and k == 3) else "\n\n \t\n".join(cur) + "\n")
    return out


def window(pid, n):
    if n <= W:
        return 0, n
    lo = xxhash.xxh3_64_intdigest(f"window:{pid}".encode()) % (n - W + 1)
    return lo, lo + W


def mkpage(pid, title, n, ns=0, redirect=False, **kw):
    revs = [dict(ts=ts(10 * k), text=t) for k, t in enumerate(texts_for(pid, n))]
    return dict(id=pid, title=title, ns=ns, redirect=redirect, revs=revs, **kw)


def build_pages():
    P = [mkpage(100 + j, f"{'ABCDEFGHIJKL'[j]}page {j}", 5 + j) for j in range(12)]          # 12 eligible, 5-16 revisions
    P[3]["title"] = "ßtraße"                                     # a non-ASCII initial: repo "other"
    P[4]["title"] = "9 lives"
    P.append(mkpage(200, "Short", 3))                            # too few revisions
    P.append(mkpage(201, "Redirect", 8, redirect=True))
    P.append(mkpage(202, "Talk:E", 8, ns=1))
    d = mkpage(203, "Deleted inside", 6); d["revs"][2]["deleted"] = True; P.append(d)
    m = mkpage(204, "Model", 6); m["revs"][4]["model"] = "json"; P.append(m)
    o = mkpage(205, "Order", 6); o["revs"][3]["ts"] = ts(1); P.append(o)
    long = mkpage(206, "Long deleted outside", 30)
    lo, hi = window(206, 30)
    long["revs"][lo - 1 if lo > 0 else hi]["deleted"] = True    # outside the window: still eligible
    P.append(long)
    ms = mkpage(207, "Multislot", 7)
    for r in ms["revs"]:
        r["slot"] = '{"x":1}'
    P.append(ms)
    nm = mkpage(208, "No model inside", 6); nm["revs"][1]["nomodel"] = True; P.append(nm)     # no <model> in the window: excluded
    no = mkpage(209, "No model outside", 30)
    lo, hi = window(209, 30)
    no["revs"][lo - 1 if lo > 0 else hi]["nomodel"] = True      # outside the window: still eligible
    P.append(no)
    return P


ELIGIBLE = [100 + j for j in range(12)] + [206, 207, 209]


def write_dumps(d, P, stub_name="stub.xml.gz", hist_name="hist.xml.bz2", hist_pages=None):
    with gzip.open(f"{d}/{stub_name}", "wb") as f:
        f.write(dump(P, with_text=False))
    with bz2.open(f"{d}/{hist_name}", "wb") as f:
        f.write(dump(hist_pages if hist_pages is not None else P, with_text=True))


def convert(d, out, *extra, stdin=None, stub="stub.xml.gz", hist="hist.xml.bz2"):
    args = [sys.executable, f"{HERE}/wiki_convert.py", "--stub", f"{d}/{stub}", "--history", "-" if stdin is not None else f"{d}/{hist}",
            "--out", out, "--seeds", "1,2", "--pages", "5", "--min-revs", str(MIN), "--max-revs", str(W), *extra]
    return subprocess.run(args, input=stdin, capture_output=True)


def expect_fail(r, msg, out):
    assert r.returncode != 0, f"{msg}: converter succeeded"
    leftovers = [x for x in os.listdir(out)] if os.path.isdir(out) else []
    assert not [x for x in leftovers if x.startswith(".wiki.staging") or re.fullmatch(r"wiki_s\d+", x)], (msg, leftovers)


def tables(o):
    return {f: pq.read_table(f"{o}/{f}") for f in ("spans.parquet", "traces.parquet", "manifest.parquet", "wiki/wide.parquet")}


def check_sample(o, s, P):
    byid = {p["id"]: p for p in P}
    hs = sorted(ELIGIBLE, key=lambda p: xxhash.xxh3_64_intdigest(f"{s}:{p}".encode()))[:5]
    order = [p["id"] for p in P if p["id"] in hs]                # dump order
    T = tables(o)
    for name, schema in (("spans.parquet", SPANS), ("traces.parquet", TRACES), ("manifest.parquet", MANIFEST), ("wiki/wide.parquet", WIDE)):
        assert T[name].schema.remove_metadata().equals(schema), (name, T[name].schema)
    man, tr, sp, wide = (T[k].to_pylist() for k in ("manifest.parquet", "traces.parquet", "spans.parquet", "wiki/wide.parquet"))
    assert [m["page_id"] for m in man] == order and [m["trace_id"] for m in man] == list(range(5))
    assert pq.ParquetFile(f"{o}/wiki/wide.parquet").num_row_groups == 5
    span = 0
    for k, pid in enumerate(order):
        p, m, t = byid[pid], man[k], tr[k]
        lo, hi = window(pid, len(p["revs"]))
        assert (m["window_lo"], m["window_hi"], m["page_revisions"], m["seed"]) == (lo, hi, len(p["revs"]), s)
        assert (m["ts_first"], m["ts_last"]) == (p["revs"][lo]["ts"], p["revs"][hi - 1]["ts"]) and m["sample_hash"] == xxhash.xxh3_64_intdigest(f"{s}:{pid}".encode())
        want = [b"[" + b",".join(orjson.dumps({"role": "block", "text": x}, option=orjson.OPT_SORT_KEYS)
                                 for x in (q.strip() for q in re.split(r"\n[ \t]*\n", r["text"])) if x) + b"]" for r in p["revs"][lo:hi]]
        got = [w["input"].encode() for w in wide[span:span + hi - lo]]
        assert got == want, pid
        assert [x["span_id"] for x in wide[span:span + hi - lo]] == list(range(span, span + hi - lo))
        assert all(x["trace_id"] == k for x in sp[span:span + hi - lo]) and [x["span_seq"] for x in sp[span:span + hi - lo]] == list(range(hi - lo))
        assert (t["trace_id"], t["span_lo"], t["span_hi"], t["n_msgs"], t["repo"], t["exit_status"], t["resolved"]) == \
               (k, span, span + hi - lo - 1, len(orjson.loads(want[-1])), repo_of(p["title"]), "", -1)
        assert b"x\\\":1" not in b"".join(got)                   # the other slot's text never appears
        span += hi - lo
    st = json.load(open(f"{o}/wiki_stats.json"))
    assert st["pass1"]["excluded"] == {"text_deleted": 1, "model": 2, "timestamp_order": 1}, st["pass1"]
    assert st["pass1"]["eligible"] == len(ELIGIBLE) and st["spans"] == span and st["traces"] == 5 and st["page_ids"] == order
    sh = st["shape"]
    assert sh["identical"] + sh["append_only"] + sh["other_change"] == sh["snapshots"] - st["traces"] and sh["snapshots"] == span
    return order


def ingest_check(d, o, s):
    sp = pq.read_table(f"{o}/spans.parquet")
    w = pq.ParquetFile(f"{o}/wiki/wide.parquet")
    dic = Dictionary(f"{d}/dict{s}.bin"); dic_f = Dictionary(f"{d}/dictf{s}.bin")
    ing_d, ing_f, rep = Ingester(dic), Ingester(dic_f), KeyReplay()
    seq_of, trace_of, empty = sp["span_seq"].to_pylist(), sp["trace_id"].to_pylist(), 0
    for g in range(w.num_row_groups):
        t = w.read_row_group(g)
        for span, text in zip(t["span_id"].to_pylist(), t["input"].to_pylist()):
            b = text.encode()
            raw = [orjson.dumps(m, option=orjson.OPT_SORT_KEYS) for m in orjson.loads(b)]
            assert b"[" + b",".join(raw) + b"]" == b, "request not canonical"
            seq_d, ids_d, *_, trans, _, _, _ = ing_d.request_text(trace_of[span], b)
            seq_f, ids_f, *_ = ing_f.request(trace_of[span], raw)
            assert seq_d == seq_f == seq_of[span] and ids_d == ids_f, "delta parsing differs from full parsing"
            ctx = rep.apply(trace_of[span], trans)
            assert b"[" + b",".join(dic.read(x) for x in ctx) + b"]" == b, "replay does not reproduce the request"
            empty += b == b"[]"
    os.close(dic.fd); os.close(dic_f.fd)
    return empty


def main():
    assert blocks("a\n\nb \n \t\n\nc\n") == ["a", "b", "c"] and blocks("") == [] and blocks("x\ny") == ["x\ny"] and blocks(" \n\n ") == []
    assert request([]) == b"[]" and orjson.loads(request(["p1", "p2"])) == [{"role": "block", "text": "p1"}, {"role": "block", "text": "p2"}]
    assert [repo_of(x) for x in ("alpha", "9x", "ßa", "Éa", "", "[[x")] == ["A", "9", "other", "other", "other", "other"]
    sh = shape([["a"], ["a", "b"], ["b", "a"], ["a", "a"], [], ["a"]])
    assert sh == dict(snapshots=6, empty=1, with_duplicate_block=1, identical=0, append_only=2, other_change=3, reentry=1), sh   # [] -> [a] is an append
    P = build_pages()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        write_dumps(d, P)
        r = convert(d, f"{d}/out")
        assert r.returncode == 0, r.stderr.decode()[-2000:]
        assert sorted(os.listdir(f"{d}/out")) == ["wiki_s1", "wiki_s2"]
        orders = [check_sample(f"{d}/out/wiki_s{s}", s, P) for s in (1, 2)]
        assert orders[0] != orders[1], "the two seeds drew the same sample"
        # standard input gives the same output
        r = convert(d, f"{d}/out_stdin", stdin=bz2.open(f"{d}/hist.xml.bz2").read())
        assert r.returncode == 0, r.stderr.decode()[-2000:]
        for s in (1, 2):
            a, b = tables(f"{d}/out/wiki_s{s}"), tables(f"{d}/out_stdin/wiki_s{s}")
            assert all(a[k].equals(b[k]) for k in a), s
            sa, sb = (json.load(open(f"{x}/wiki_s{s}/wiki_stats.json")) for x in (f"{d}/out", f"{d}/out_stdin"))
            assert {k: v for k, v in sa.items() if k not in ("seconds", "src")} == {k: v for k, v in sb.items() if k not in ("seconds", "src")}
        # the ingestion path and the baselines
        empties = 0
        for s in (1, 2):
            empties += ingest_check(d, f"{d}/out/wiki_s{s}", s)
        assert empties > 0, "no empty snapshot reached the ingestion check"
        r = subprocess.run([sys.executable, f"{HERE}/ingest_run.py", "--data", f"{d}/out/wiki_s1", "--managers", "wiki", "--out", f"{d}/ing1",
                            "--work", d, "--interleave", "4"], capture_output=True)
        assert r.returncode == 0, r.stderr.decode()[-2000:]
        arr = pq.read_table(f"{d}/ing1/wiki/arrays.parquet")
        assert arr.num_rows == pq.read_metadata(f"{d}/out/wiki_s1/spans.parquet").num_rows, "one array row per span"
        r = subprocess.run([sys.executable, f"{HERE}/ingest_ablate.py", "--data", f"{d}/out/wiki_s1", "--manager", "wiki", "--work", f"{d}/abl",
                            "--out", f"{d}/abl.json", "--interleave", "4"], capture_output=True)
        assert r.returncode == 0, r.stderr.decode()[-2000:]
        res = json.load(open(f"{d}/abl.json"))
        assert all(v["wrong"] == 0 for v in res["variants"].values() if v["must_be_exact"]), res["variants"]

        # failures: nothing is left behind, and a later correct run succeeds
        sampled = sorted(set(orders[0]) | set(orders[1]))
        f = f"{d}/fail"
        write_dumps(d, P, hist_name="h_missing.xml.bz2", hist_pages=[p for p in P if p["id"] != sampled[0]])
        expect_fail(convert(d, f, hist="h_missing.xml.bz2"), "sampled page missing", f)
        extra = json.loads(json.dumps(P)); x = next(p for p in extra if p["id"] == sampled[1]); x["revs"].append(dict(ts=ts(9999), text="More."))
        write_dumps(d, P, hist_name="h_count.xml.bz2", hist_pages=extra)
        expect_fail(convert(d, f, hist="h_count.xml.bz2"), "revision count mismatch", f)
        shifted = json.loads(json.dumps(P)); x = next(p for p in shifted if p["id"] == sampled[2])
        lo, hi = window(x["id"], len(x["revs"])); x["revs"][hi - 1]["ts"] = ts(10 * hi + 5)
        write_dumps(d, P, hist_name="h_ts.xml.bz2", hist_pages=shifted)
        expect_fail(convert(d, f, hist="h_ts.xml.bz2"), "window timestamps differ", f)
        write_dumps(d, P + [mkpage(100, "Dup", 6)], stub_name="s_dup.xml.gz", hist_name="h_dup.xml.bz2")
        expect_fail(convert(d, f, stub="s_dup.xml.gz", hist="h_dup.xml.bz2"), "repeated page id", f)
        expect_fail(convert(d, f, "--pages", str(len(ELIGIBLE) + 1)), "too few eligible pages", f)
        for bad in (["--seeds", "1,1"], ["--pages", "0"], ["--name", "a/b"], ["--seeds", "x"]):
            expect_fail(convert(d, f, *bad), f"bad arguments {bad}", f)
        os.makedirs(f"{d}/taken/wiki_s2"); open(f"{d}/taken/wiki_s2/keep", "w").write("x")
        r = convert(d, f"{d}/taken")
        assert r.returncode != 0 and os.listdir(f"{d}/taken") == ["wiki_s2"] and os.listdir(f"{d}/taken/wiki_s2") == ["keep"]
        r = convert(d, f)
        assert r.returncode == 0 and sorted(os.listdir(f)) == ["wiki_s1", "wiki_s2"], r.stderr.decode()[-2000:]
        for s in (1, 2):
            a, b = tables(f"{f}/wiki_s{s}"), tables(f"{d}/out/wiki_s{s}")
            assert all(a[k].equals(b[k]) for k in a), s
    print("TEST_WIKI_CONVERT_OK")


if __name__ == "__main__":
    main()
