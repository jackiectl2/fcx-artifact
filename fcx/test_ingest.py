"""Property test of the differencer: replaying its transitions must reproduce every request, in order, for random edit
sequences (appends, drops at the front and in the middle, in-place replacements, middle insertions, duplicates, full rewrites)
and for any edit budget. Mutations of the differencer must be caught, otherwise the test has no resolution."""
import os, random

import numpy as np

import ingest


def run(make, n_traces=400, seed=0, budget=64):
    rng, fails = random.Random(seed), 0
    for _ in range(n_traces):
        d = make(budget)
        ctx, active, fresh = [], {}, 1000
        for step in range(rng.randint(1, 40)):
            r = rng.random()
            if r < 0.05:                                          # full rewrite
                ctx = [rng.randint(0, 50) for _ in range(rng.randint(0, 20))]
            else:
                if r < 0.35 and ctx:                              # drop from the front or the middle
                    i = rng.randint(0, len(ctx) - 1); del ctx[i:i + rng.randint(1, 4)]
                if rng.random() < 0.3 and ctx:                    # replace in place (elision)
                    ctx[rng.randint(0, len(ctx) - 1)] = rng.randint(0, 5)
                if rng.random() < 0.2 and ctx:                    # insert in the middle
                    ctx.insert(rng.randint(0, len(ctx)), rng.randint(0, 50))
                for _ in range(rng.randint(0, 3)):                # append; small IDs repeat, so duplicates occur
                    fresh += 1; ctx.append(rng.choice([fresh, rng.randint(0, 5)]))
            out, ck = d.feed(ctx)
            ok = True
            for op, occ, mid, key in out:                         # an independent replay of the log
                if op == 1:
                    ok = ok and occ not in active; active[occ] = (key, mid)
                else:
                    ok = ok and active.pop(occ, None) == (key, mid)
            keys = sorted(active.values())
            ok = ok and [m for _, m in keys] == ctx and len({k for k, _ in keys}) == len(keys)
            if ck is not None:
                ok = ok and ck[2].tolist() == ctx and sorted(ck[0].tolist()) == sorted(active) and ck[1].tolist() == [k for k, _ in keys]
            if not ok:
                fails += 1; break
    return fails


class NoMidpoint(ingest.Differencer):                             # mutation: new occurrences get keys after the last one, wherever they go
    def feed(self, ids):
        out, ck = super().feed(ids)
        top = max(self.ord) if self.ord else 0
        return [(op, occ, mid, top + 1 + i if op == 1 else key) for i, (op, occ, mid, key) in enumerate(out)], ck


class ForgetClose(ingest.Differencer):                            # mutation: the log misses the closes
    def feed(self, ids):
        out, ck = super().feed(ids)
        return [t for t in out if t[0] == 1], ck


def run_online(n_traces=60, seed=0, hash_bytes=16):
    """Interleaved arrival: the ingester must produce, per trace, the transitions it produces when the traces arrive one after
    the other (message IDs differ, their structure does not), and replaying them must reproduce every request byte for byte.
    Returns the number of traces that fail."""
    rng = random.Random(seed)
    traces = []
    for t in range(n_traces):
        ctx, reqs = [], []
        for step in range(rng.randint(1, 25)):
            if rng.random() < 0.3 and ctx:
                i = rng.randint(0, len(ctx) - 1); del ctx[i:i + rng.randint(1, 3)]
            ctx += [b"m%d" % rng.randint(0, 40) if rng.random() < 0.3 else b"t%d-%d-%d" % (t, step, j) for j in range(rng.randint(0, 3))]
            reqs.append(list(ctx))
        traces.append(reqs)
    order = [t for t, reqs in enumerate(traces) for _ in reqs]
    logs = []
    for arrival in (order, rng.sample(order, len(order))):       # contiguous, then interleaved
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fcx_test_dict.bin")
        dic = ingest.Dictionary(path)
        if hash_bytes < 16:                                       # mutation: a weak hash, so that different messages collide
            real, xx = ingest.xxhash.xxh3_128_digest, ingest.xxhash
            ingest.xxhash = type("weak", (), {"xxh3_128_digest": staticmethod(lambda b: real(b)[:hash_bytes])})
        ing, text, log, nxt = ingest.Ingester(dic, ckpt=4), {}, [[] for _ in traces], [0] * n_traces
        try:
            for t in arrival:
                seq, ids, new, variants, trans, ck = ing.request(t, traces[t][nxt[t]])
                assert seq == nxt[t]
                nxt[t] += 1; text.update(new); log[t].append(trans)
        finally:
            if hash_bytes < 16:
                ingest.xxhash = xx
            os.close(dic.fd); os.remove(path)                    # the dictionary is not read after this point
        for t in range(n_traces):
            ing.end(t)
        assert not ing.active
        logs.append((log, text))
    fails = 0
    for t, reqs in enumerate(traces):
        ok = [[(op, occ, key) for op, occ, _, key in tr] for tr in logs[0][0][t]] == [[(op, occ, key) for op, occ, _, key in tr] for tr in logs[1][0][t]]
        for log, text in logs:
            active = {}
            for req, trans in zip(reqs, log[t]):
                for op, occ, mid, key in trans:
                    if op == 1:
                        active[occ] = (key, mid)
                    else:
                        del active[occ]
                ok = ok and [text[m] for _, m in sorted(active.values())] == req
        fails += not ok
    return fails


def variant_matches(needle, a, p, s, ins, wv, wb, use_wb=True):
    """The rule of query.matched_variants in Python: decide whether a variant contains the needle from its base's answer and
    the two windows; build the variant only when the windows cannot decide. Returns (answer, whether the variant was built)."""
    if needle in wv:
        return True, False
    if not use_wb or needle not in wb:
        return needle in a, False
    return needle in a[:p] + ins + a[len(a) - s:], True


def run_variants(n=3000, seed=0, context=None, use_wb=True):
    """Random bases, random small edits, random needles of up to K + 1 bytes: the rule must agree with a search in the variant.
    Returns (wrong answers, variants built)."""
    rng, wrong, built = random.Random(seed), 0, 0
    k = ingest.K if context is None else context
    for _ in range(n):
        a = bytes(rng.choice(b"abc") for _ in range(rng.randint(ingest.MIN_VARIANT, ingest.MIN_VARIANT + 400)))
        i = rng.choice([0, len(a), rng.randint(0, len(a))])     # edits at both ends and inside
        j = min(len(a), i + rng.randint(0, 40))
        b = a[:i] + bytes(rng.choice(b"abcd") for _ in range(rng.randint(0, 40))) + a[j:]
        ps = ingest.splice(a, b)
        assert ps is not None and b == a[:ps[0]] + b[ps[0]:len(b) - ps[1]] + a[len(a) - ps[1]:]
        p, s = ps
        wv, wb = b[max(0, p - k):len(b) - s + k], a[max(0, p - k):len(a) - s + k]
        assert context is not None or (wv, wb) == ingest.windows(a, b, p, s)
        for _ in range(20):
            src = rng.choice([a, b]); L = rng.randint(1, ingest.K + 1); o = rng.randint(0, len(src) - L)
            needle = src[o:o + L] if rng.random() < 0.8 else bytes(rng.choice(b"abcd") for _ in range(rng.randint(1, 6)))
            got, mat = variant_matches(needle, a, p, s, b[p:len(b) - s], wv, wb, use_wb)
            wrong += got != (needle in b); built += mat
    return wrong, built


def run_text(n_traces=300, seed=0):
    """Delta ingestion (request_text) against plain ingestion (request) on random request sequences: appends, drops at the
    front and in the middle, replacements by near-copies (a role, a cache mark or the tail of the text changes), insertions,
    duplicates, rewrites, repeats. Both must give the same message IDs, transitions and checkpoints for every request.
    Returns (traces that differ, bytes parsed by the delta path, bytes of all requests)."""
    import orjson
    rng, fails, parsed, total, fresh = random.Random(seed), 0, 0, 0, 0
    dump = lambda m: orjson.dumps(m, option=orjson.OPT_SORT_KEYS)

    def message(like=None):
        nonlocal fresh
        if like is not None and rng.random() < 0.7:               # a near-copy of an existing message
            m = orjson.loads(like)
            r = rng.random()
            if r < 0.3:
                m["role"] = "tool" if m["role"] == "user" else "user"
            elif r < 0.6:
                m["cache_control"] = {"type": "ephemeral"}
            elif isinstance(m["content"], str):
                m["content"] = m["content"][:-1] + rng.choice("xyz")
            else:
                m["content"][-1]["text"] += "!"
            return dump(m)
        fresh += 1
        text = "shared head " * rng.choice([0, 1, 8]) + "body %d " % fresh + "pad " * rng.randint(0, 40)
        content = text if rng.random() < 0.6 else [{"type": "text", "text": text}, {"type": "text", "text": "second block %d" % rng.randint(0, 3)}]
        return dump({"role": rng.choice(["user", "assistant", "tool"]), "content": content})

    for t in range(n_traces):
        A, B = (ingest.Ingester(ingest.Dictionary(os.path.join(os.environ.get("TMPDIR", "/tmp"), "fcx_test_%s.bin" % c)), ckpt=4) for c in "ab")
        ctx, ok = [message(), message()], True
        for step in range(rng.randint(1, 30)):
            r = rng.random()
            if r < 0.05:
                ctx = [message() for _ in range(rng.randint(0, 6))]
            elif r < 0.10:
                pass                                              # the same request again
            else:
                if rng.random() < 0.3 and len(ctx) > 2:           # drop after the head or in the middle
                    i = rng.choice([1, rng.randint(1, len(ctx) - 1)]); del ctx[i:i + rng.randint(1, 3)]
                if rng.random() < 0.4 and ctx:                    # replace by a near-copy
                    i = rng.randint(0, len(ctx) - 1); ctx[i] = message(ctx[i])
                if rng.random() < 0.2 and ctx:                    # insert in the middle
                    ctx.insert(rng.randint(0, len(ctx)), message(rng.choice(ctx)))
                if rng.random() < 0.1 and ctx:                    # duplicate
                    ctx.append(rng.choice(ctx))
                ctx += [message() for _ in range(rng.randint(0, 3))]
            text = b"[" + b",".join(ctx) + b"]"
            a, b = A.request(t, list(ctx)), B.request_text(t, text)
            same = a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3] and a[4] == b[4]
            same = same and (a[5] is None) == (b[5] is None) and (a[5] is None or all((u == v).all() for u, v in zip(a[5], b[5])))
            ok = ok and same
            parsed += b[7]; total += len(text)
        fails += not ok
        for ing, c in ((A, "a"), (B, "b")):
            os.close(ing.dict.fd); os.remove(os.path.join(os.environ.get("TMPDIR", "/tmp"), "fcx_test_%s.bin" % c))
    return fails, parsed, total


if __name__ == "__main__":
    fails, parsed, total = run_text()
    assert fails == 0, fails
    def loose_recurring(text, pos, pmv, pends, k, m):            # mutation: a message counts as unchanged although its last 8 bytes differ
        off, j = pos - (pends[k - 1] + 1 if k else 1), 0
        while k + j < m:
            st = pends[k + j - 1] + 1 if k + j else 1
            if not text.startswith(pmv[st:pends[k + j] - 8], st + off):
                break
            j += 1
        return j
    real, ingest.recurring = ingest.recurring, loose_recurring
    loose = run_text()[0]
    ingest.recurring = real
    assert loose > 0, loose
    print("delta ingestion: 0 of 300 traces differ; %.0f%% of the bytes parsed; the mutation makes %d traces differ" % (100 * parsed / total, loose))
    wrong, built = run_variants()
    assert wrong == 0, wrong
    no_wb, short = run_variants(use_wb=False)[0], run_variants(context=8)[0]      # mutations: skip the base window; too little context
    assert no_wb > 0 and short > 0, (no_wb, short)
    assert ingest.splice(b"x" * 2000, b"y" * 2000) is None and ingest.splice(b"x" * 2000, b"x" * 1000) is None
    print("variant rule: 0 wrong of 60000; variants built in", built, "cases; mutations give", no_wb, "and", short, "wrong answers")
    assert run_online() == 0
    try:                                                          # with a one-byte hash the byte comparison must stop the run
        run_online(hash_bytes=1); raise SystemExit("a hash collision went unnoticed")
    except ValueError as e:
        print("collision caught:", e)
    for budget in (0, 2, 64):
        assert run(lambda b: ingest.Differencer(budget=b, ckpt=4), budget=budget) == 0, budget
    small_gap, ingest.GAP = ingest.GAP, 4                         # tiny gaps force the relabelling path
    assert run(lambda b: ingest.Differencer(budget=b, ckpt=4)) == 0
    ingest.GAP = small_gap
    caught = run(lambda b: NoMidpoint(budget=b)), run(lambda b: ForgetClose(budget=b))
    assert min(caught) > 0, caught
    d = ingest.Differencer(budget=0); [d.feed(list(range(k))) for k in range(1, 30)]
    assert d.fallbacks == 0                                       # pure appends never need the fallback
    print("ok; mutations caught in", caught, "of 400 traces")
