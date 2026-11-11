"""Property tests for the E-ING variants: the exact variants must reproduce every context for random edit sequences
(appends, drops, in-place replacements, middle insertions, duplicates, full rewrites); the lossy set difference must be exact
on append-only contexts without duplicates and must fail once duplicates occur. A mutation of the no-key differencer must be
caught."""
import random

import os

import orjson

from ingest import Dictionary
from ingest_ablate import ArrayIngester, KeyReplay, NoKeyDifferencer, derive, replayer

KINDS = ("hash_array", "pos_diff", "set_diff", "mset_lifo")


def check_baselines(traces):
    """Number of traces on which each baseline's replayed context differs from the true context at some call."""
    fails = dict.fromkeys(KINDS, 0)
    for t, seq in enumerate(traces):
        for kind in KINDS:
            state, rep = {}, replayer(kind)
            for ctx in seq:
                rows, _ = derive(kind, state, t, ctx)
                if rep.apply(t, rows) != ctx:
                    fails[kind] += 1; break
    return fails


def append_only(rng, n_traces=200):
    for _ in range(n_traces):
        ctx, fresh, seq = [], 0, []
        for _ in range(rng.randint(1, 30)):
            for _ in range(rng.randint(0, 3)):
                fresh += 1; ctx.append(fresh)
            seq.append(list(ctx))
        yield seq


def edits(rng, n_traces=300):
    for _ in range(n_traces):
        ctx, fresh, seq = [], 1000, []
        for _ in range(rng.randint(1, 40)):
            r = rng.random()
            if r < 0.05:
                ctx = [rng.randint(0, 50) for _ in range(rng.randint(0, 20))]
            else:
                if r < 0.35 and ctx:
                    i = rng.randint(0, len(ctx) - 1); del ctx[i:i + rng.randint(1, 4)]
                if rng.random() < 0.3 and ctx:
                    ctx[rng.randint(0, len(ctx) - 1)] = rng.randint(0, 5)
                if rng.random() < 0.2 and ctx:
                    ctx.insert(rng.randint(0, len(ctx)), rng.randint(0, 50))
                for _ in range(rng.randint(0, 3)):
                    fresh += 1; ctx.append(rng.choice([fresh, rng.randint(0, 5)]))
            seq.append(list(ctx))
        yield seq


def check_array_delta(cls=ArrayIngester, n_traces=200, seed=11):
    """hash_array_delta: interleaved random request streams (drops, near-copies, insertions, duplicates, rewrites) through the
    delta parser; every call's array, read back through the dictionary, must rebuild the request byte for byte."""
    rng, dump, fresh = random.Random(seed), (lambda m: orjson.dumps(m, option=orjson.OPT_SORT_KEYS)), [0]

    def message(like=None):
        if like is not None and rng.random() < 0.6:
            m = orjson.loads(like); m["content"] = m["content"][:-1] + rng.choice("xyz")
            return dump(m)
        fresh[0] += 1
        return dump({"content": "body %d " % fresh[0] + "pad " * rng.randint(0, 50), "role": rng.choice(["user", "tool"])})

    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fcx_test_arrdelta.bin")
    dic = Dictionary(path); ing = cls(dic, 4)
    seqs = []
    for t in range(n_traces):
        ctx, seq = [message()], []
        for _ in range(rng.randint(1, 20)):
            r = rng.random()
            if r < 0.05:
                ctx = [message() for _ in range(rng.randint(0, 5))]
            else:
                if rng.random() < 0.3 and len(ctx) > 1:
                    del ctx[rng.randint(0, len(ctx) - 1)]
                if rng.random() < 0.3 and ctx:
                    i = rng.randint(0, len(ctx) - 1); ctx[i] = message(ctx[i])
                if rng.random() < 0.2 and ctx:
                    ctx.insert(rng.randint(0, len(ctx)), rng.choice(ctx))
                ctx += [message() for _ in range(rng.randint(0, 3))]
            seq.append(b"[" + b",".join(ctx) + b"]")
        seqs.append(seq)
    pos, fails, live = [0] * n_traces, set(), [t for t in range(n_traces)]
    while live:
        t = rng.choice(live)
        text = seqs[t][pos[t]]
        call, ids, _, _, arr, ck, _, _ = ing.request_text(t, text)
        if call != pos[t] or ck is not None or arr.tolist() != list(ids) or b"[" + b",".join(dic.read(x) for x in arr.tolist()) + b"]" != text:
            fails.add(t)
        pos[t] += 1
        if pos[t] == len(seqs[t]):
            ing.end(t); live.remove(t)
    os.close(dic.fd); os.remove(path)
    return len(fails)


class DropLast(ArrayIngester):                                    # mutation: the array loses its last message
    def _diff(self, trace, ids, new):
        c, ids, new, v, arr, ck = super()._diff(trace, ids, new)
        return c, ids, new, v, arr[:-1], ck


def check_nokey(cls, budget):
    fails = 0
    for seq in edits(random.Random(budget)):
        d, rep = cls(budget), KeyReplay()
        for ctx in seq:
            rows, _ = d.feed(ctx)
            try:
                ok = rep.apply(0, rows) == ctx
            except AssertionError:
                ok = False
            if not ok:
                fails += 1; break
    return fails


class ShiftOnly(NoKeyDifferencer):                                # mutation: forgets to reopen kept occurrences that moved
    def feed(self, ids):
        old = self.ids.tolist()
        rows, ck = super().feed(ids)
        moved = {(k, m) for op, k, m, _ in rows if op == -1 and m in ids}
        return [r for r in rows if not (r[0] == -1 and (r[1], r[2]) in moved) and not (r[0] == 1 and r[2] in old)], ck


def main():
    for b in (0, 2, 64):
        f = check_nokey(NoKeyDifferencer, b)
        assert f == 0, f"no-key differencer failed on {f} traces (budget {b})"
    assert check_nokey(ShiftOnly, 64) > 0, "the test does not catch a broken no-key differencer"
    f = check_baselines(edits(random.Random(7)))
    assert f["hash_array"] == 0 and f["pos_diff"] == 0, f
    assert f["set_diff"] > 0 and f["mset_lifo"] > 0, f"lossy baselines never failed on edits with duplicates: {f}"
    g = check_baselines(append_only(random.Random(3)))
    assert all(v == 0 for v in g.values()), f"baselines must be exact on append-only contexts without duplicates: {g}"
    assert check_array_delta() == 0, "hash_array_delta does not rebuild every request"
    assert check_array_delta(DropLast) > 0, "the test does not catch a broken hash_array_delta"
    print("TEST_ABLATE_OK", f, g, "hash_array_delta exact; mutation caught")


if __name__ == "__main__":
    main()
