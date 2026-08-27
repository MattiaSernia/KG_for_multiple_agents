"""
Plot the effect of the CHUNK DIMENSION on the experiments -- line/bar-chart
style, exactly like plot/plot_injection.py.

X axis  : chunk dimension (0 -> 10 -> 30 -> 50 -> 100).
Y axis  : % of satisfied requirements (col 1) / constraints (col 2).
Series  : one single line for the frozen configuration
          llama / TURTLE / TEXT, at a fixed KG-injection point (default
          C_{OAP}).  C_null is drawn as a dashed baseline.
One row is produced per CFT, plus a group-average figure (mean +/- std across
CFTs, with --media) and the corresponding token-usage figures.

Note: the chunk sweep (10/30/50/100) currently exists on disk only for the
C_{OA} injection point.  For C_{OAP} only chunk 0 is present, so the other x
points stay empty until those experiments are run -- the script fills them in
automatically once the folders appear.

Usage examples
--------------
    python3 plot_chunk/plot_chunk.py                     # all cfts, C_{OAP}
    python3 plot_chunk/plot_chunk.py --cft belval        # single cft
    python3 plot_chunk/plot_chunk.py --injection C_{OA}  # full data sanity check
    python3 plot_chunk/plot_chunk.py --media             # average across cfts
"""
import os
import re
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = os.path.join(ROOT, "experiments")
HERE        = os.path.dirname(os.path.abspath(__file__))

CFT_META = {
    "belval":  {"total_req": 23, "total_con": 19},
    "chrb":    {"total_req": 25, "total_con": 23},
    "cabinet": {"total_req": 27, "total_con": 26},
}

REL_PATH = "llama/TURTLE/TEXT"

MODE_VALFILE = {
    "C_{O}":   "single_validation_kg.txt",
    "C_{OA}":  "single_validation_kgagents.txt",
    "C_{OP}":  "single_validation_kgcft.txt",
    "C_{OAP}": "single_validation_kgagents_tri.txt",
}

CHUNKS = [0, 10, 30, 50, 100]

LINE_COLOR = "#1565C0"

parser = argparse.ArgumentParser()
parser.add_argument("--cft", choices=list(CFT_META.keys()))
parser.add_argument("--injection", choices=list(MODE_VALFILE.keys()),
                    default="C_{OAP}",
                    help="KG injection point to freeze (default C_{OAP})")
parser.add_argument("--media", action="store_true",
                    help="Average across all CFTs instead of one row per CFT")
args = parser.parse_args()

MODE        = args.injection
VALFILE     = MODE_VALFILE[MODE]
active_cfts = [args.cft] if args.cft else list(CFT_META.keys())


def read_scores(cft, chunk):
    """Return (%req, %con) for the frozen config at one chunk dim, or (None, None)."""
    path = os.path.join(EXPERIMENTS, cft, REL_PATH, f"{MODE}_{chunk}", VALFILE)
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    req = len(re.findall(r'^REQ-\d+', content, re.MULTILINE))
    con = len(re.findall(r'^CON-\d+', content, re.MULTILINE))
    return (req / CFT_META[cft]["total_req"] * 100,
            con / CFT_META[cft]["total_con"] * 100)


def read_tokens(cft, chunk):
    path = os.path.join(EXPERIMENTS, cft, REL_PATH, f"{MODE}_{chunk}", "tokens.json")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["prompt_tokens"], data["completion_tokens"]


def read_null_scores(cft):
    path = os.path.join(EXPERIMENTS, cft, "C_null", "C_null_0",
                        "single_validation_nokg.txt")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    req = len(re.findall(r'^REQ-\d+', content, re.MULTILINE))
    con = len(re.findall(r'^CON-\d+', content, re.MULTILINE))
    return (req / CFT_META[cft]["total_req"] * 100,
            con / CFT_META[cft]["total_con"] * 100)


def read_null_tokens(cft):
    path = os.path.join(EXPERIMENTS, cft, "C_null", "C_null_0", "tokens.json")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["prompt_tokens"], data["completion_tokens"]


def nan(v):
    return v if v is not None else np.nan


score_req  = {cft: [] for cft in active_cfts}   
score_con  = {cft: [] for cft in active_cfts}
tok_prompt = {cft: [] for cft in active_cfts}
tok_compl  = {cft: [] for cft in active_cfts}

for cft in active_cfts:
    for chunk in CHUNKS:
        r, c = read_scores(cft, chunk)
        p, o = read_tokens(cft, chunk)
        score_req[cft].append(nan(r)); score_con[cft].append(nan(c))
        tok_prompt[cft].append(nan(p)); tok_compl[cft].append(nan(o))

null_scores = {cft: read_null_scores(cft) for cft in active_cfts}
null_tokens = {cft: read_null_tokens(cft) for cft in active_cfts}

x = np.arange(len(CHUNKS))
chunk_ticklabels = [str(c) for c in CHUNKS]

cft_tag   = f"_{args.cft}" if args.cft else ""
media_tag = "_media" if args.media else ""
inj_tag   = re.sub(r'[^A-Za-z]', '', MODE)   


def mean_ignore_none(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def matrix_media(mat_by_cft):
    """Average a [chunk] vector across all active CFTs -> [chunk]."""
    stacked = np.array([mat_by_cft[cft] for cft in active_cfts], dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def matrix_std(mat_by_cft):
    stacked = np.array([mat_by_cft[cft] for cft in active_cfts], dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanstd(stacked, axis=0)


def fmt_val(v, is_token):
    return f"{v/1000:.0f}k" if is_token else f"{v:.0f}%"


def line_panel(ax, heights, std, title, ylabel, null_val, is_token):
    """heights, std: [chunk] arrays.  std may be None (per-CFT, no error bars)."""
    heights = np.array(heights, dtype=float)
    errs = None if std is None else np.nan_to_num(np.array(std, dtype=float))
    ax.errorbar(x, heights, yerr=errs, marker="o", markersize=5,
                linewidth=2, linestyle="-", color=LINE_COLOR,
                ecolor="0.4", capsize=3, label="llama/TURTLE/TEXT")
    for xi, h in enumerate(heights):
        if np.isfinite(h) and h > 0:
            txt = fmt_val(h, is_token)
            if std is not None and np.isfinite(std[xi]):
                txt += f"\n±{fmt_val(std[xi], is_token).lstrip()}"
            ax.text(x[xi], h, txt, ha="center", va="bottom", fontsize=6)
    if null_val is not None:
        lab = f"$C_{{null}}$ ({fmt_val(null_val, is_token)})"
        ax.axhline(null_val, color="black", linestyle="--", linewidth=1.8,
                   alpha=0.85, label=lab)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(chunk_ticklabels, fontsize=11)
    ax.set_xlim(-0.3, len(CHUNKS) - 0.7)
    ax.set_xlabel("Chunk dimension", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if is_token:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    else:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 107)


def make_line_figure(metric_specs, suptitle, outname):
    n_cols = len(metric_specs)
    if args.media:
        fig = plt.figure(figsize=(max(9, n_cols * 6.5), 5))
        gs  = fig.add_gridspec(1, n_cols, wspace=0.28)
        for ci, (data, title, ylabel, nidx, is_tok) in enumerate(metric_specs):
            ax = fig.add_subplot(gs[0, ci])
            null_val = mean_ignore_none(
                [(null_tokens if is_tok else null_scores)[cft][nidx] for cft in active_cfts])
            line_panel(ax, matrix_media(data), matrix_std(data),
                       f"{title} (avg over CFTs)", ylabel, null_val, is_tok)
    else:
        n_rows = len(active_cfts)
        fig = plt.figure(figsize=(max(9, n_cols * 6.5), n_rows * 4.2))
        gs  = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.28)
        for ri, cft in enumerate(active_cfts):
            for ci, (data, title, ylabel, nidx, is_tok) in enumerate(metric_specs):
                ax = fig.add_subplot(gs[ri, ci])
                null_val = (null_tokens if is_tok else null_scores)[cft][nidx]
                line_panel(ax, data[cft], None,
                           f"{cft.upper()} {title}", ylabel, null_val, is_tok)

    handles, labels_leg = fig.axes[0].get_legend_handles_labels()
    leg_y = -0.12 if (args.media or len(active_cfts) == 1) else -0.03
    fig.legend(handles, labels_leg, fontsize=9, ncol=min(len(handles), 6),
               loc="lower center", bbox_to_anchor=(0.5, leg_y))
    fig.suptitle(suptitle + (" [avg over CFTs]" if args.media else ""), fontsize=14)
    out = os.path.join(HERE, outname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out}")


make_line_figure(
    [
        (score_req, "Satisfied Requirements", "% satisfied", 0, False),
        (score_con, "Satisfied Constraints",  "% satisfied", 1, False),
    ],
    f"Chunk dimension vs satisfied req/con (llama/TURTLE/TEXT, ${MODE}$)",
    f"plot_chunk_{inj_tag}{cft_tag}{media_tag}.png",
)

make_line_figure(
    [
        (tok_prompt, "Input Tokens (prompt)",      "tokens", 0, True),
        (tok_compl,  "Output Tokens (completion)", "tokens", 1, True),
    ],
    f"Chunk dimension vs token usage (llama/TURTLE/TEXT, ${MODE}$)",
    f"plot_chunk_tokens_{inj_tag}{cft_tag}{media_tag}.png",
)

plt.show()
