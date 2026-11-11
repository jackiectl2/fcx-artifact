"""Test of figures/fast_summary.py on fixtures built from the real v1 results (copied into a temporary results tree): v2 files
are the v1 files plus ingest_impl "v2" and a hash_array_delta variant with the arrays of hash_array and a chosen CPU time.
Correct fixtures give status ok with the expected ratios; each damage gives blocked (or incomplete for a missing file).
usage: .venv/bin/python figures/test_fast_summary.py"""
import copy, json, os, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fast_summary as F                                          # noqa: E402

REAL = HERE.parent / "results"


def build(tmp, damage=None):
    for part in F.V2:
        for data, manager in F.streams(part):
            for r2, r1 in zip(F.V2[part], F.V1[part]):
                src = REAL / r1 / f"{data}__{manager}.json"
                (tmp / r1).mkdir(exist_ok=True); (tmp / r2).mkdir(exist_ok=True)
                shutil.copy(src, tmp / r1 / src.name)
                d = json.load(open(src)); d["ingest_impl"] = "v2"
                hd = copy.deepcopy(d["variants"]["hash_array"]); hd["cpu_s"] = d["variants"]["fcx_b64"]["cpu_s"] * 0.5
                hd["parsed_bytes"] = d["variants"]["fcx_b64"]["parsed_bytes"]
                d["variants"]["hash_array_delta"] = hd
                if damage and damage[0] == (r2, data, manager) and damage[1] is not None:
                    damage[1](d)
                if not (damage and damage[0] == (r2, data, manager) and damage[1] is None):
                    json.dump(d, open(tmp / r2 / src.name, "w"))
    (tmp / "online_v2").mkdir(); (tmp / "online_v1").mkdir(exist_ok=True)
    for p in (REAL / "online_v1").glob("*.json"):
        d = json.load(open(p)); d["ingest_impl"] = "v2"; json.dump(d, open(tmp / "online_v2" / p.name, "w"))
    (tmp / "fast_cmp_v2").mkdir()                                 # ingest_cmp fixtures: the earlier project's runs (read-only) + ingest_impl v2
    for c, run in (("swea", "r1_2026-10-03_late"), ("smith", "r1_2026-10-03_late"), ("oh", "r1_2026-10-03_late"), ("gym", "r1_2026-10-24")):
        src = REAL / run / f"cap_{c}_s1__ingest_cmp.json"            # in the package; elsewhere set FCX_OLD_RESULTS to the earlier results
        if not src.exists():
            src = Path(os.environ["FCX_OLD_RESULTS"]) / run / src.name
        d = json.load(open(src)); d["ingest_impl"] = "v2"
        if damage and damage[0] == ("fast_cmp_v2", c):
            damage[1](d)
        json.dump(d, open(tmp / "fast_cmp_v2" / f"cap_{c}_s1__ingest_cmp.json", "w"))
    for s in (1, 2, 3):
        (tmp / "wiki" / f"wiki_s{s}").mkdir(parents=True, exist_ok=True)
        shutil.copy(REAL / "wiki" / f"wiki_s{s}" / "convert_stats.json", tmp / "wiki" / f"wiki_s{s}" / "convert_stats.json")


def status(damage=None):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        F.RES = F.ing.RES = F.ods.RES = tmp
        build(tmp, damage)
        sys.argv = ["x", "--out", str(tmp / "out.json")]
        try:
            F.main()
        except SystemExit:
            pass
        return json.load(open(tmp / "out.json"))


ok = status()
assert ok["status"] == "ok", (ok.get("problems"), ok.get("missing"), ok.get("online", {}).get("problems"))
assert ok["files"] == 17 * 3 + 34 + 12 + 3, ok["files"]
assert ok["wiki"]["hash_delta_cpu"]["single_range"] == [0.5, 0.5], ok["wiki"]
assert ok["swe"]["s1"]["append"]["hash_delta_cpu"]["median_range"] == [0.5, 0.5]
key = ("ing_s1_v2", "cap_swea_s1", "prefix")
damages = {
    "non-timing field differs from v1": (key, lambda d: d["variants"]["fcx_b64"].update(records=d["variants"]["fcx_b64"]["records"] + 1)),
    "wrong implementation": (("ing_chat_v2", "cap_chat_s2", "autogen_ht"), lambda d: d.update(ingest_impl="v1")),
    "hash_array_delta wrong": (("ing_wiki_v2", "wiki_s3", "wiki"), lambda d: d["variants"]["hash_array_delta"].update(wrong=1)),
    "hash_array_delta other arrays": (("ing_s23_v2", "cap_oh_s3", "prefix"), lambda d: d["variants"]["hash_array_delta"].update(array_elems=1)),
    "hash_array_delta missing": (("ing_s1_v2r3", "cap_gym_s1", "autogen_ht"), lambda d: d["variants"].pop("hash_array_delta")),
    "hash_array_delta other parse": (("ing_s1_v2", "cap_oh_s1", "lc_trim32k"), lambda d: d["variants"]["hash_array_delta"].update(parsed_bytes=1)),
    "hash_array_delta slice wrong": (("ing_chat_v2", "cap_chat_s1", "prefix"), lambda d: d["variants"]["hash_array_delta"]["by_slice"]["dup"].update(wrong=1)),
    "hash_array_delta zero time": (("ing_s23_v2", "cap_swea_s2", "prefix"), lambda d: d["variants"]["hash_array_delta"].update(wall_s=0.0)),
}
for name, dm in damages.items():
    st = status(dm)["status"]
    assert st == "blocked", (name, st)
c = ok["ingest_cmp"]
assert c["status"] == "ok" and c["streams"] == 17 and c["files_compared"] == [14, 17], c
# the fixtures are the earlier runs: the validator must reproduce the ranges the paper quotes for them
assert [round(x, 1) for x in c["full_mb_per_s"]] == [61.4, 119.6] and [round(x, 1) for x in c["delta_mb_per_s"]] == [82.7, 291.2], c
shares = [r["parsed_share"] for r in c["per_stream"]]
assert c["parsed_share"] == [0.0275, 0.0839] and [round(min(shares), 7), round(max(shares), 7)] == [0.0275437, 0.0838996], c["parsed_share"]


def swap(d):                                                        # the same managers in another order
    for k in ("delta0", "delta1"):
        items = list(d[k]["policies"].items())
        d[k]["policies"] = dict(items[1:] + items[:1])


cmp_damages = {"a table differs": (("fast_cmp_v2", "oh"), lambda d: d.update(files_that_differ=["messages"])),
               "v1 implementation": (("fast_cmp_v2", "gym"), lambda d: d.update(ingest_impl="v1")),
               "a manager missing": (("fast_cmp_v2", "smith"), lambda d: d["delta1"]["policies"].pop("lc_trim32k")),
               "managers reordered": (("fast_cmp_v2", "swea"), swap),
               "wrong source": (("fast_cmp_v2", "gym"), lambda d: d["delta0"].update(src="data/cap_gym_s2")),
               "file count": (("fast_cmp_v2", "oh"), lambda d: d.update(files_compared=13)),
               "non-finite throughput": (("fast_cmp_v2", "smith"), lambda d: d["delta1"]["policies"]["prefix"]["ingest"].update(logical_mb_per_s=float("inf"))),
               "full parse skipped bytes": (("fast_cmp_v2", "swea"), lambda d: d["delta0"]["policies"]["autogen_ht"].update(parsed_bytes=1))}
for name, dm in cmp_damages.items():
    assert status(dm)["ingest_cmp"]["status"] == "blocked", name
assert status((key, None))["status"] == "incomplete"
print("TEST_FAST_SUMMARY_OK: ok on", ok["files"], "files;", len(damages), "damages blocked; a missing file is incomplete;",
      "ingest_cmp ok on 17 streams (full %s MB/s, delta %s MB/s) and %d damages blocked" % (c["full_mb_per_s"], c["delta_mb_per_s"], len(cmp_damages)))
