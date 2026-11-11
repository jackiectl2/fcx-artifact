"""Plan P of stage_fallback in bench.py, as logic: on random calls it must find exactly the calls whose serialized request holds
the needle, whenever its guard holds. Calls include empty and single-message ones, repeats, commas inside messages and non-ASCII text."""
import random


def serialized(ids, text):
    return "[" + ",".join(text[i] for i in ids) + "]"


def plan_p(n, ids, text):
    if any(n in text[i] for i in ids):
        return True
    return any(i > 0 and c == "," and any(text[x].endswith(n[:i]) and text[y].startswith(n[i + 1:]) for x, y in zip(ids, ids[1:])) for i, c in enumerate(n))


def guard(n, text):
    commas = [i for i, c in enumerate(n) if c == ","]
    pieces = {n[i + 1:j] for i in commas for j in commas if j > i}
    return n and n[0] not in "[," and not n.endswith("]") and not (pieces & set(text))


def main(rounds=200000, seed=7):
    rng = random.Random(seed)
    alphabet = "ab,é]["
    checked = used = 0
    for _ in range(rounds):
        text = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 5))) for _ in range(5)]
        ids = [rng.randrange(5) for _ in range(rng.randint(0, 4))]
        s = serialized(ids, text)
        n = s[rng.randint(0, len(s)):][:rng.randint(1, 8)] if rng.random() < 0.8 else "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
        checked += 1
        if guard(n, text):
            used += 1
            if plan_p(n, ids, text) != (n in s):
                raise SystemExit(f"wrong: needle {n!r}, messages {[text[i] for i in ids]!r}")
    print(f"plan P: 0 wrong of {used} needles that pass the guard ({checked} drawn)")


if __name__ == "__main__":
    main()
