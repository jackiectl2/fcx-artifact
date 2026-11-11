"""Shared paths and plotting style of every figure, table and summary script in this package.

Paths are package-relative: inputs under results/, outputs under paper/figures/.

Two layout profiles. The profile is chosen by the name of the script being run; FCX_STYLE_PROFILE=ieee or v1 overrides it.
  ieee  IEEEtran conference widths (column 3.5 in, text 7.16 in), Times-first serif fonts, corpus labels SWE-rebench and
        SWE-Gym. Used by fig7_ingest_baselines.py and by every script not named in V1_SCRIPTS.
  v1    column 3.33 in, text 7.0 in, Libertine-first serif fonts, corpus labels "SWE-rebench OH" and "SWE-Gym OH". Figures 2-6
        of the paper were drawn with these settings (LaTeX scales the PDFs to the page), so their scripts use this profile to
        reproduce the PDFs in the paper.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LATE = RESULTS / "r1_2026-10-03_late"
OUT = ROOT / "paper" / "figures"

V1_SCRIPTS = {"fig2_workload", "fig3_c2", "fig4_encodings", "fig5_ingest", "fig6_supplement",
              "tab2_exact", "tab3_cost", "tab4_managers", "mixed_index_summary"}


def _profile():
    chosen = os.environ.get("FCX_STYLE_PROFILE", "")
    if chosen:
        if chosen not in ("ieee", "v1"):
            raise SystemExit(f"FCX_STYLE_PROFILE must be ieee or v1, not {chosen!r}")
        return chosen
    main = getattr(sys.modules.get("__main__"), "__file__", None) or ""
    return "v1" if Path(main).stem in V1_SCRIPTS else "ieee"


PROFILE = _profile()

# Okabe-Ito
COLORS = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
          "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}
MARK = {"swea": "o", "smith": "s", "oh": "D", "gym": "^"}   # corpus marker, so that corpora stay apart in greyscale
MANAGERS = {"prefix": "append only", "swea_last5": "SWE-agent last 5 obs.", "swea_window": "SWE-agent closed window",
            "swea_cache2": "SWE-agent cache marks", "autogen_ht": "AutoGen head+tail", "autogen_buf32": "AutoGen buffer 32",
            "lc_trim32k": "LangChain trim 32k"}

if PROFILE == "v1":
    COLUMN_IN = 3.33
    TEXT_IN = 7.0
    CORPUS = {"swea": ("SWE-agent", COLORS["blue"]), "oh": ("SWE-rebench OH", COLORS["red"]),
              "smith": ("SWE-smith", COLORS["green"]), "gym": ("SWE-Gym OH", COLORS["purple"])}
    SERIF = ["Linux Libertine O", "Libertinus Serif", "Times New Roman", "DejaVu Serif"]
else:
    COLUMN_IN = 3.5
    TEXT_IN = 7.16
    CORPUS = {"swea": ("SWE-agent", COLORS["blue"]), "smith": ("SWE-smith", COLORS["green"]),
              "oh": ("SWE-rebench", COLORS["red"]), "gym": ("SWE-Gym", COLORS["purple"])}
    SERIF = ["Times New Roman", "Times", "DejaVu Serif"]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": SERIF,
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "lines.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def ink(color, k=0.35):
    """A darker shade of a palette colour, for text drawn in that colour (pale text is hard to read in print)."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(v * (1 - k)) for v in (r, g, b))


def load(path):
    return json.load(open(path))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
