"""FCX operator API: the SQL of every plan for a context query over one sealed segment in ClickHouse.

A segment holds the message dictionary `msg`, the trace table `traces` (a trace is the span-ID range span_lo..span_hi) and,
per context policy, tables named in T (layout -> table name):
  d, ckpt   the transition log and its checkpoints (source of truth)
  i         Interval(msg_id, trace_id, occ, ord, span_from, span_to)   one row per message occurrence
  a         SpanCtx(span_id, msg_ids)                                  the context of every span, in order
  nf, ni    ContextItem(span_id, pos, msg_id) sorted by span and by message: the conventional normalized form (baseline)
  b, a_bf, a_tx   further baselines: bitmap postings, arrays with host indexes

Operators (each returns plan name -> SQL; the benchmark driver times exactly these statements):
  ctx_any(T, M, F)      spans whose context holds a message of M; optional trace-level filter F
  ctx_count(T, M)       per span, the number of context messages in M
  ctx_sum(T, val, M)    per span, the sum of a per-message value over the context messages in M (all messages if M is None)
  ctx_items(T)          the ordered context of the spans in table `targets` (reconstruction)
M is a subquery returning message IDs; matched() builds it from a predicate on the dictionary, which is therefore evaluated
once per distinct message; matched_variants() does the same over a dictionary that stores small-edit variants as splices.
"""


def matched(pred, msg="msg"):
    return f"(SELECT msg_id FROM {msg} WHERE {pred})"


def matched_variants(pred, bases, var, vb):
    """Like matched(), over a dictionary that stores a small-edit variant as a splice of its base message instead of in full.
    Tables: `bases` (msg_id, text) with the base messages; `var` (msg_id, base_id, p, s, ins, wv, wb): the variant is
    base[:p] + ins + base[len - s:], and wv / wb are the bytes of the variant / of the base around the edit with K bytes of
    context; Join table `vb`: base_id -> vars, the variants of a base as tuples (msg_id, p, s, ins, wb, wv).
    pred(col) is the SQL of a substring predicate of at most K + 1 bytes on the text expression `col`.
    A variant contains the substring if its window does. Otherwise it agrees with its base, unless the base's window contains
    the substring: then the occurrence may be the one the edit removed, and the variant is rebuilt and searched.
    The predicate therefore runs once per base message and on the windows."""
    built = "concat(substring(text, 1, x.2), x.4, substring(text, length(text) - x.3 + 1, x.3))"
    keep = f"arrayFilter(x -> ({pred('x.6')}) OR NOT ({pred('x.5')}) OR ({pred(built)}), joinGet('{vb}', 'vars', msg_id))"
    return (f"(SELECT arrayJoin(arrayConcat([msg_id], arrayMap(x -> x.1, {keep}))) AS msg_id FROM {bases} WHERE {pred('text')} "
            f"UNION ALL SELECT msg_id FROM {var} WHERE {pred('wv')})")


def expand(lo, hi):
    return f"groupBitmapState(toUInt32(arrayJoin(range({lo}, toUInt32({hi} + 1)))))"


def slots(frm, extra=""):
    """One row per (interval, span) pair."""
    return f"SELECT toUInt32(arrayJoin(range(span_from, toUInt32(span_to + 1)))) AS span_id{extra} FROM {frm}"


def runs(d, where="1"):
    """Membership runs replayed from the transition log: pair each open with its close; a run without a close ends with its trace."""
    return (f"(SELECT g.trace_id AS trace_id, g.occ AS occ, g.o AS ord, g.m AS msg_id, g.f AS span_from, toUInt32(if(g.nc > 0, g.c - 1, tr.span_hi)) AS span_to "
            f"FROM (SELECT trace_id, occ, any(ord) AS o, any(msg_id) AS m, minIf(span_id, op = 1) AS f, minIf(span_id, op = -1) AS c, countIf(op = -1) AS nc "
            f"FROM {d} WHERE {where} GROUP BY trace_id, occ) AS g INNER JOIN traces AS tr ON tr.trace_id = g.trace_id)")


def merged(hits):
    """Union of the intervals in `hits` per trace: a sweep that starts a new run when an interval begins past everything before it.
    `occ` makes the sort order total. Without it the two window passes may order equal intervals differently, and an interval
    flagged as the start of a run can then land in the previous run (found by the set check on window32)."""
    order = "PARTITION BY trace_id ORDER BY span_from, span_to, occ"
    return (f"(SELECT min(span_from) AS s, max(span_to) AS e FROM (SELECT trace_id, span_from, span_to, "
            f"sum(is_new) OVER ({order} ROWS UNBOUNDED PRECEDING) AS grp FROM ("
            f"SELECT trace_id, span_from, span_to, occ, span_from > 1 + max(span_to) OVER ({order} "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS is_new FROM {hits})) GROUP BY trace_id, grp)")


def ctx_any(T, M, F, n_matched):
    """Plans as name -> (aggregate that yields the span-ID bitmap, FROM clause).
    F: predicate on `traces` (a trace-level filter), or None; n_matched: size of M (one baseline is quadratic in it)."""
    P = {}
    rng = "toUInt32(arrayJoin(range(span_lo, toUInt32(span_hi + 1))))"
    cbm = f"(SELECT groupBitmapState({rng}) FROM traces WHERE {F})"   # candidate spans
    ctr = f"(SELECT trace_id FROM traces WHERE {F})"                  # candidate traces
    fa = f" AND span_id IN (SELECT {rng} FROM traces WHERE {F})" if F else ""

    def bitmap(name, agg, frm):                                   # membership first: intersect the candidate bitmap last
        P[name] = (f"bitmapAnd({agg}, {cbm})" if F else agg, frm)

    if "a" in T:
        P["A.exists"] = ("groupBitmapState(span_id)", f"{T['a']} WHERE arrayExists(x -> x IN {M}, msg_ids){fa}")
        P["A.unnest"] = ("groupBitmapState(span_id)", f"{T['a']} ARRAY JOIN msg_ids AS m WHERE m IN {M}{fa}")
    if "a_bf" in T and n_matched <= 20000:                        # hasAny is quadratic in the matched set
        P["A.bloom"] = ("groupBitmapState(span_id)", f"{T['a_bf']} WHERE hasAny(msg_ids, (SELECT groupArray(msg_id) FROM {M})){fa}")
    if "a_tx" in T:
        P["A.textidx"] = ("groupBitmapState(span_id)", f"{T['a_tx']} WHERE hasAny(arrayMap(x -> toString(x), msg_ids), (SELECT groupArray(toString(msg_id)) FROM {M})){fa}")
    if "i" in T:
        hits = f"{T['i']} WHERE msg_id IN {M}"
        bitmap("I.expand", expand("span_from", "span_to"), hits)
        bitmap("I.sweep", expand("s", "e"), merged(hits))
        if F:                                                     # metadata first; the runs of a candidate trace lie inside it
            P["I.pruned"] = (expand("span_from", "span_to"), f"{hits} AND trace_id IN {ctr}")
            P["I.pruned-sweep"] = (expand("s", "e"), merged(f"{hits} AND trace_id IN {ctr}"))
            P["I.join"] = ("groupBitmapState(c.span_id)", f"(SELECT trace_id, span_from, span_to FROM {hits}) AS m INNER JOIN (SELECT {rng} AS span_id, trace_id FROM traces WHERE {F}) AS c "
                                                         f"ON c.trace_id = m.trace_id AND c.span_id >= m.span_from AND c.span_id <= m.span_to")
    if "ni" in T:                                                 # normalized membership, sorted by message
        bitmap("N.inv", "groupBitmapState(span_id)", f"{T['ni']} WHERE msg_id IN {M}")
    if "nf" in T:                                                 # normalized membership, sorted by span
        P["N.fwd"] = ("groupBitmapState(span_id)", f"{T['nf']} WHERE msg_id IN {M}{fa}")
    if "b" in T:
        bitmap("B.or", "groupBitmapOrState(spans)", f"{T['b']} WHERE msg_id IN {M}")
        if F:                                                     # metadata first: restrict every posting before the union
            P["B.and-or"] = (f"groupBitmapOrState(bitmapAnd(spans, {cbm}))", f"{T['b']} WHERE msg_id IN {M}")
    if "d" in T:
        bitmap("D.replay", expand("span_from", "span_to"), runs(T["d"], f"msg_id IN {M}"))
        if F:
            P["D.pruned"] = (expand("span_from", "span_to"), runs(T["d"], f"msg_id IN {M} AND trace_id IN {ctr}"))
    return P


def route(F_is_single_trace):
    """The plan the system runs by default: arrays when the filter names one trace, intervals otherwise."""
    return "A.exists" if F_is_single_trace else "I.expand"


def ctx_count(T, M):
    """Plans as name -> SELECT span_id, v. Sparse output: a span without a matching message is omitted and its count is 0."""
    Q = {}
    if "a" in T:
        Q["A"] = f"SELECT span_id, toInt64(arrayCount(x -> x IN {M}, msg_ids)) AS v FROM {T['a']} WHERE v > 0"
    if "i" in T:
        Q["I.expand"] = f"SELECT span_id, toInt64(count()) AS v FROM ({slots(T['i'] + f' WHERE msg_id IN {M}')}) GROUP BY span_id"
    if "d" in T:
        Q["D.replay"] = f"SELECT span_id, toInt64(count()) AS v FROM ({slots(runs(T['d'], f'msg_id IN {M}'))}) GROUP BY span_id"
    if "ni" in T:
        Q["N.inv"] = f"SELECT span_id, toInt64(count()) AS v FROM {T['ni']} WHERE msg_id IN {M} GROUP BY span_id"
    return Q


def ctx_sum(T, val, M=None):
    """Plans as name -> SELECT span_id, v. val(col) is the SQL of the per-message value for the message ID in `col`.
    With M the sum runs over the context messages in M; without it over the whole context (D.sweep then needs table `spans`).
    Sparse output: a plan may omit a span whose sum is 0, or return it with v = 0; an omitted span's sum is 0."""
    Q = {}
    if M is not None:
        if "a" in T:
            Q["A"] = f"SELECT span_id, toInt64(arraySum(arrayMap(x -> if(x IN {M}, {val('x')}, 0), msg_ids))) AS v FROM {T['a']} WHERE v != 0"
        if "i" in T:
            Q["I.expand"] = f"SELECT span_id, toInt64(sum(n)) AS v FROM ({slots(T['i'] + f' WHERE msg_id IN {M}', ', ' + val('msg_id') + ' AS n')}) GROUP BY span_id"
        if "ni" in T:
            Q["N.inv"] = f"SELECT span_id, toInt64(sum({val('msg_id')})) AS v FROM {T['ni']} WHERE msg_id IN {M} GROUP BY span_id"
        return Q
    if "a" in T:
        Q["A"] = f"SELECT span_id, toInt64(arraySum(arrayMap(x -> {val('x')}, msg_ids))) AS v FROM {T['a']}"
    if "i" in T:
        Q["I.expand"] = f"SELECT span_id, toInt64(sum(n)) AS v FROM ({slots(T['i'], ', ' + val('msg_id') + ' AS n')}) GROUP BY span_id"
    if "d" in T:
        Q["D.sweep"] = (f"SELECT span_id, sum(delta) OVER (PARTITION BY trace_id ORDER BY span_id ROWS UNBOUNDED PRECEDING) AS v FROM (SELECT s.trace_id AS trace_id, s.span_id AS span_id, "
                        f"sum(toInt64(d.op) * toInt64({val('d.msg_id')})) AS delta FROM spans AS s LEFT JOIN {T['d']} AS d ON d.span_id = s.span_id GROUP BY s.trace_id, s.span_id)")
    if "nf" in T:
        Q["N.fwd"] = f"SELECT span_id, toInt64(sum({val('msg_id')})) AS v FROM {T['nf']} GROUP BY span_id"
    return Q


def ctx_items(T):
    """Ordered context (message IDs) of the spans in `targets`: read the array; replay the log from a checkpoint; collect the
    covering intervals; read the normalized rows."""
    ordered = "arrayMap(x -> x.2, arraySort(groupArray((o, msg_id))))"
    Q = {}
    if "a" in T:
        Q["A"] = f"SELECT span_id, msg_ids FROM {T['a']} WHERE span_id IN (SELECT span_id FROM targets) ORDER BY span_id"
    if "d" in T:
        Q["D"] = (f"SELECT s, {ordered} FROM (SELECT s, occ, o, msg_id FROM ("
                  f"SELECT s, occ, o, msg_id, toInt8(1) AS op FROM (SELECT t.span_id AS s, c.occs AS occs, c.ords AS ords, c.msg_ids AS msg_ids FROM {T['ckpt']} AS c INNER JOIN targets AS t "
                  f"ON t.ck = c.span_id WHERE c.span_id IN (SELECT ck FROM targets)) ARRAY JOIN occs AS occ, ords AS o, msg_ids AS msg_id "
                  f"UNION ALL SELECT t.span_id AS s, d.occ AS occ, d.ord AS o, d.msg_id AS msg_id, d.op AS op FROM {T['d']} AS d INNER JOIN targets AS t ON t.trace_id = d.trace_id "
                  f"WHERE d.trace_id IN (SELECT trace_id FROM targets) AND d.span_id > t.ck AND d.span_id <= t.span_id) "
                  f"GROUP BY s, occ, o, msg_id HAVING sum(op) > 0) GROUP BY s ORDER BY s")
    if "i" in T:
        Q["I"] = (f"SELECT t.span_id AS s, arrayMap(x -> x.2, arraySort(groupArray((i.ord, i.msg_id)))) FROM {T['i']} AS i INNER JOIN targets AS t ON t.trace_id = i.trace_id "
                  f"WHERE t.span_id >= i.span_from AND t.span_id <= i.span_to GROUP BY s ORDER BY s")
    if "nf" in T:
        Q["N"] = (f"SELECT span_id, arrayMap(x -> x.2, arraySort(groupArray((pos, msg_id)))) FROM {T['nf']} WHERE span_id IN (SELECT span_id FROM targets) "
                  f"GROUP BY span_id ORDER BY span_id")
    return Q


def ctx_text(ids, text_of):
    """The serialized request of a span from its ordered message IDs: a JSON array of the logged messages.
    text_of(col) is the SQL that returns the stored text of the message ID in `col`."""
    return f"concat('[', arrayStringConcat(arrayMap(x -> {text_of('x')}, {ids}), ','), ']')"
