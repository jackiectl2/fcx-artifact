"""Context policies: the messages an agent sends at each LLM call of a recorded trajectory.

Plain, direct code on purpose: the requests produced here are the reference that the ingested log is checked against.
A context is a list of items; an item is a message position, or ("stub", position) for a tool output that was elided.

  prefix          every earlier message
  window<N>       the head (everything up to the first user message) plus the last N other messages
  amortized<N>    like window<N>, but messages are dropped only once N of them may go (the context swings between N and 2N)
  elide<K>        every earlier message, with all but the last K tool outputs replaced by a short stub
"""
import re


def head_len(roles):
    """Number of leading messages that are never dropped: up to the first user message, else the leading tools/system ones."""
    for i, r in enumerate(roles):
        if r == "user":
            return i + 1
    return next((i for i, r in enumerate(roles) if r not in ("tools", "system")), len(roles))


def contexts(roles, policy):
    """Yield the context of every LLM call (one per assistant message), in call order."""
    kind, n = re.fullmatch(r"([a-z]+)(\d*)", policy).groups()
    n, h = int(n or 0), head_len(roles)
    frontier = h                                                  # amortized: first retained position after the head
    for k, r in enumerate(roles):
        if r != "assistant":
            continue
        seen = list(range(k))
        if kind == "prefix":
            yield seen
        elif kind == "window":
            yield seen[:h] + seen[h:][-n:]
        elif kind == "amortized":
            if max(h, k - n) - frontier >= n:
                frontier = max(h, k - n)
            yield seen[:h] + [j for j in seen[h:] if j >= frontier]
        elif kind == "elide":
            tools = [j for j in seen if roles[j] == "tool"]
            old = set(tools[:max(0, len(tools) - n)])
            yield [("stub", j) if j in old else j for j in seen]
        else:
            raise ValueError(policy)


def stub_text(n_chars):
    return '{"content":"[tool output elided: %d characters]","role":"tool"}' % n_chars
