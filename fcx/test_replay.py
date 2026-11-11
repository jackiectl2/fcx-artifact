"""The policies in replay.py against the M0 simulation (prefix, window) and against their definitions (amortized, elide)."""
import random

import m0_convert, replay

if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(2000):
        roles = ["tools"] + [rng.choice(["system", "user", "assistant", "assistant", "tool", "tool"]) for _ in range(rng.randint(1, 80))]
        if rng.random() < 0.5:
            roles[1:3] = ["system", "user"][: len(roles) - 1]
        h = replay.head_len(roles)
        for n in (1, 5, 32):
            assert list(replay.contexts(roles, f"window{n}")) == m0_convert.simulate(roles, n)
            win = list(replay.contexts(roles, f"window{n}"))
            for ctx, w, k in zip(replay.contexts(roles, f"amortized{n}"), win, [i for i, r in enumerate(roles) if r == "assistant"]):
                tail = ctx[min(h, k):]                             # a superset of the window, contiguous up to the call, shorter than 2N
                assert set(w) <= set(ctx) and ctx[: min(h, k)] == list(range(min(h, k))) and tail == list(range(k - len(tail), k)) and len(tail) < 2 * n
            for ctx, k in zip(replay.contexts(roles, f"elide{n}"), [i for i, r in enumerate(roles) if r == "assistant"]):
                pos = [x[1] if isinstance(x, tuple) else x for x in ctx]
                kept_tools = [x for x in ctx if not isinstance(x, tuple) and roles[x] == "tool"]
                assert pos == list(range(k)) and all(roles[x[1]] == "tool" for x in ctx if isinstance(x, tuple))
                assert kept_tools == [j for j in range(k) if roles[j] == "tool"][-n:]
        assert list(replay.contexts(roles, "prefix")) == m0_convert.simulate(roles, None)
    print("ok")
