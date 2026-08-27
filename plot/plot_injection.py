"""
Plot the effect of the KG-injection point on the experiments -- bar-chart
style, exactly like plot.py.

X axis  : where the KG is injected (C_{O} -> C_{OA} / C_{OP} -> C_{OAP}).
Y axis  : % of satisfied requirements (col 1) / constraints (col 2).
Bars    : one grouped bar per combination of the *other* (non-frozen)
          parameters. The format is fixed to TURTLE (turtle-light), so it is
          NOT a free variable here; the free variables are extractor, schema
          and text (any of them can be frozen from the CLI).
One row is produced per CFT, plus a group-average figure (bars grouped by
dimension, mean +/- std) and the corresponding token-usage figures, exactly
like plot.py.  C_null is drawn as a dashed baseline.

Usage examples
--------------
    python3 plot/plot_injection.py                    # all cfts, all combos
    python3 plot/plot_injection.py --cft belval       # single cft
    python3 plot/plot_injection.py --extractor phi4   # freeze the extractor
    python3 plot/plot_injection.py --text TEXT        # freeze the text flag
    python3 plot/plot_injection.py --media            # average across cfts
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

MODES = ["C_{O}", "C_{OA}", "C_{OP}", "C_{OAP}"]
MODE_LABELS = {m: f"${m}$" for m in MODES}
MODE_VALFILE = {
    "C_{O}":   "single_validation_kg.txt",
    "C_{OA}":  "single_validation_kgagents.txt",
    "C_{OP}":  "single_validation_kgcft.txt",
    "C_{OAP}": "single_validation_kgagents_tri.txt",
}

INJ_CONFIGS = [
    ("phi4",  "SCHEMA", "TEXT",    "phi4/SCHEMA/TURTLE/TEXT"),
    ("phi4",  "SCHEMA", "NO_TEXT", "phi4/SCHEMA/TURTLE/NO_TEXT"),
    ("llama", None,     "TEXT",    "llama/TURTLE/TEXT"),
    ("llama", None,     "NO_TEXT", "llama/TURTLE/NO_TEXT"),
]

DIM_ORDER  = ["extractor", "schema", "text", "injection"]
DIM_VALUES = {
    "extractor": ["phi4",   "llama"],
    "schema":    ["SCHEMA", "NO_SCHEMA"],
    "text":      ["TEXT",   "NO_TEXT"],
    "injection": MODES,
}
DIM_SHORT = {
    "extractor": {"phi4": "M_2",  "llama": "M_1"},
    "schema":    {"SCHEMA": "S",  "NO_SCHEMA": "NS"},
    "text":      {"TEXT":   "T",  "NO_TEXT":   "NT"},
    "injection": {"C_{O}": "O", "C_{OA}": "OA", "C_{OP}": "OP", "C_{OAP}": "OAP"},
}

PALETTE = ["#1565C0", "#E65100", "#2E7D32", "#9C27B0",
           "#00695C", "#C62828", "#F9A825", "#4527A0"]
CFT_COLORS   = ["#1565C0", "#E65100", "#2E7D32", "#9C27B0"]
GROUP_COLORS = ["#5C6BC0", "#EF5350", "#43A047", "#FF8F00"]
TEXT_COLORS  = {"TEXT": "#1565C0", "NO_TEXT": "#E65100"}
LINE_STYLES  = ["-", "--", "-.", ":"]

parser = argparse.ArgumentParser()
parser.add_argument("--cft",       choices=list(CFT_META.keys()))
parser.add_argument("--extractor", choices=DIM_VALUES["extractor"])
parser.add_argument("--schema",    choices=DIM_VALUES["schema"])
parser.add_argument("--text",      choices=DIM_VALUES["text"])
parser.add_argument("--media", action="store_true",
                    help="Average across all CFTs instead of one row per CFT")
args = parser.parse_args()

freeze = {d: getattr(args, d) for d in ["extractor", "schema", "text"]
          if getattr(args, d) is not None}
active_cfts = [args.cft] if args.cft else list(CFT_META.keys())


def read_scores(rel_path, mode, cft):
    """Return (%req, %con) for one config at one injection point, or (None, None)."""
    path = os.path.join(EXPERIMENTS, cft, rel_path, f"{mode}_0", MODE_VALFILE[mode])
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    req = len(re.findall(r'^REQ-\d+', content, re.MULTILINE))
    con = len(re.findall(r'^CON-\d+', content, re.MULTILINE))
    return (req / CFT_META[cft]["total_req"] * 100,
            con / CFT_META[cft]["total_con"] * 100)


def read_tokens(rel_path, mode, cft):
    path = os.path.join(EXPERIMENTS, cft, rel_path, f"{mode}_0", "tokens.json")
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

def cfg_val(cfg, dim):
    ext, sch, txt, _ = cfg
    return {"extractor": ext, "schema": sch, "text": txt}[dim]


def matches_freeze(cfg):
    for d, v in freeze.items():
        val = cfg_val(cfg, d)
        if val is None:            
            continue
        if val != v:
            return False
    return True


series_free = [d for d in ["extractor", "schema", "text"] if d not in freeze]


def sort_key(cfg):
    return tuple(
        DIM_VALUES[d].index(cfg_val(cfg, d)) if cfg_val(cfg, d) is not None else 999
        for d in series_free
    )


filtered = sorted([c for c in INJ_CONFIGS if matches_freeze(c)], key=sort_key)
if not filtered:
    raise SystemExit("No configurations match the selected filters.")


def cfg_label(cfg):
    parts = [DIM_SHORT[d][cfg_val(cfg, d)] for d in series_free if cfg_val(cfg, d) is not None]
    return "$" + "/".join(parts) + "$" if parts else "all"


labels = [cfg_label(cfg) for cfg in filtered]

def nan(v):
    return v if v is not None else np.nan


score_req  = {cft: [] for cft in active_cfts}   
score_con  = {cft: [] for cft in active_cfts}
tok_prompt = {cft: [] for cft in active_cfts}
tok_compl  = {cft: [] for cft in active_cfts}

for cft in active_cfts:
    for cfg in filtered:
        rel_path = cfg[-1]
        r_row, c_row, p_row, o_row = [], [], [], []
        for mode in MODES:
            r, c = read_scores(rel_path, mode, cft)
            p, o = read_tokens(rel_path, mode, cft)
            r_row.append(nan(r)); c_row.append(nan(c))
            p_row.append(nan(p)); o_row.append(nan(o))
        score_req[cft].append(r_row); score_con[cft].append(c_row)
        tok_prompt[cft].append(p_row); tok_compl[cft].append(o_row)

null_scores = {cft: read_null_scores(cft) for cft in active_cfts}
null_tokens = {cft: read_null_tokens(cft) for cft in active_cfts}

x = np.arange(len(MODES))
mode_ticklabels = [MODE_LABELS[m] for m in MODES]

freeze_str = ", ".join(f"{d}={v}" for d, v in freeze.items()) if freeze else "no freeze"
freeze_tag = "_".join(freeze.values()) if freeze else "all"
cft_tag    = f"_{args.cft}" if args.cft else ""
media_tag  = "_media" if args.media else ""


def mean_ignore_none(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def matrix_media(mat_by_cft):
    """Average a [cfg][mode] matrix across all active CFTs -> [cfg][mode]."""
    stacked = np.array([mat_by_cft[cft] for cft in active_cfts], dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def matrix_std(mat_by_cft):
    stacked = np.array([mat_by_cft[cft] for cft in active_cfts], dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanstd(stacked, axis=0)


def fmt_val(v, is_token):
    return f"{v/1000:.0f}k" if is_token else f"{v:.0f}%"


def bar_panel(ax, mat, std, title, ylabel, null_val, is_token):
    """Line chart: x = injection point, one line per config (coloured by text).

    mat, std: [cfg][mode] arrays.  std may be None (per-CFT, no error bars).
    """
    style_counter = {}
    for ci in range(len(filtered)):
        heights = np.array(mat[ci], dtype=float)
        errs = None if std is None else np.nan_to_num(np.array(std[ci], dtype=float))
        txt_val = cfg_val(filtered[ci], "text")
        color = TEXT_COLORS.get(txt_val, PALETTE[ci % len(PALETTE)])
        k = style_counter.get(txt_val, 0)
        style_counter[txt_val] = k + 1
        ls = LINE_STYLES[k % len(LINE_STYLES)]
        ax.errorbar(x, heights, yerr=errs, marker="o", markersize=5,
                    linewidth=2, linestyle=ls, color=color,
                    ecolor="0.4", capsize=3, label=labels[ci])
        for xi, h in enumerate(heights):
            if np.isfinite(h) and h > 0:
                txt = fmt_val(h, is_token)
                if std is not None and np.isfinite(std[ci][xi]):
                    txt += f"\n±{fmt_val(std[ci][xi], is_token).lstrip()}"
                ax.text(x[xi], h, txt, ha="center", va="bottom", fontsize=6)
    if null_val is not None:
        lab = f"$C_{{null}}$ ({fmt_val(null_val, is_token)})"
        ax.axhline(null_val, color="black", linestyle="--", linewidth=1.8,
                   alpha=0.85, label=lab)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_ticklabels, fontsize=11)
    ax.set_xlim(-0.3, len(MODES) - 0.7)
    ax.set_xlabel("KG injection point", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if is_token:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    else:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 107)


def make_bar_figure(metric_specs, suptitle, outname):
    n_cols = len(metric_specs)
    if args.media:
        fig = plt.figure(figsize=(max(9, n_cols * 6.5), 5))
        gs  = fig.add_gridspec(1, n_cols, wspace=0.28)
        for ci, (data, title, ylabel, nidx, is_tok) in enumerate(metric_specs):
            ax = fig.add_subplot(gs[0, ci])
            null_val = mean_ignore_none(
                [(null_tokens if is_tok else null_scores)[cft][nidx] for cft in active_cfts])
            bar_panel(ax, matrix_media(data), matrix_std(data),
                      f"{title} (avg over CFTs)", ylabel, null_val, is_tok)
    else:
        n_rows = len(active_cfts)
        fig = plt.figure(figsize=(max(9, n_cols * 6.5), n_rows * 4.2))
        gs  = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.28)
        for ri, cft in enumerate(active_cfts):
            for ci, (data, title, ylabel, nidx, is_tok) in enumerate(metric_specs):
                ax = fig.add_subplot(gs[ri, ci])
                null_val = (null_tokens if is_tok else null_scores)[cft][nidx]
                bar_panel(ax, np.array(data[cft], dtype=float), None,
                          f"{cft.upper()} {title}", ylabel, null_val, is_tok)

    handles, labels_leg = fig.axes[0].get_legend_handles_labels()
    leg_y = -0.12 if (args.media or len(active_cfts) == 1) else -0.03
    fig.legend(handles, labels_leg, fontsize=9, ncol=min(len(handles), 6),
               loc="lower center", bbox_to_anchor=(0.5, leg_y))
    fig.suptitle(suptitle + (" [avg over CFTs]" if args.media else ""), fontsize=14)
    out = os.path.join(HERE, outname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out}")


make_bar_figure(
    [
        (score_req, "Satisfied Requirements", "% satisfied", 0, False),
        (score_con, "Satisfied Constraints",  "% satisfied", 1, False),
    ],
    f"KG injection vs satisfied req/con ({freeze_str})",
    f"plot_inj_{freeze_tag}{cft_tag}{media_tag}.png",
)

make_bar_figure(
    [
        (tok_prompt, "Input Tokens (prompt)",      "tokens", 0, True),
        (tok_compl,  "Output Tokens (completion)", "tokens", 1, True),
    ],
    f"KG injection vs token usage ({freeze_str})",
    f"plot_inj_tokens_{freeze_tag}{cft_tag}{media_tag}.png",
)


cells = []   
for ci, cfg in enumerate(filtered):
    for mi, mode in enumerate(MODES):
        cells.append({
            "extractor": cfg_val(cfg, "extractor"),
            "schema":    cfg_val(cfg, "schema"),
            "text":      cfg_val(cfg, "text"),
            "injection": mode,
            "req":    {cft: score_req[cft][ci][mi]  for cft in active_cfts},
            "con":    {cft: score_con[cft][ci][mi]  for cft in active_cfts},
            "prompt": {cft: tok_prompt[cft][ci][mi] for cft in active_cfts},
            "compl":  {cft: tok_compl[cft][ci][mi]  for cft in active_cfts},
        })

group_dims = []
for dim in ["injection"] + series_free:
    present = {c[dim] for c in cells if c[dim] is not None}
    if len(present) >= 2:
        group_dims.append(dim)


def cell_values(dim, val, metric, cft):
    return [c[metric][cft] for c in cells if c[dim] == val]


def cell_values_media(dim, val, metric):
    """Per-cell mean across CFTs, collected over all matching cells."""
    out = []
    for c in cells:
        if c[dim] != val:
            continue
        vals = [c[metric][cft] for cft in active_cfts if np.isfinite(c[metric][cft])]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def group_panel(ax, metric, title, ylabel, null_val, is_token, cft=None):
    xg = np.arange(len(group_dims))
    for gi, dim in enumerate(group_dims):
        vals_present = [v for v in DIM_VALUES[dim]
                        if any(c[dim] == v for c in cells)]
        n_keys = len(vals_present)
        w = 0.8 / n_keys
        for ki, val in enumerate(vals_present):
            raw = (cell_values_media(dim, val, metric) if cft is None
                   else cell_values(dim, val, metric, cft))
            raw = [v for v in raw if np.isfinite(v)]
            if not raw:
                continue
            mean, std = float(np.mean(raw)), float(np.std(raw))
            offset = (ki - (n_keys - 1) / 2) * w
            ax.bar(xg[gi] + offset, mean, w, yerr=std, ecolor="0.4", capsize=2,
                   color=GROUP_COLORS[ki % len(GROUP_COLORS)], edgecolor="white")
            ax.text(xg[gi] + offset, mean, f"{DIM_SHORT[dim][val]}\n{fmt_val(mean, is_token)}",
                    ha="center", va="bottom", fontsize=6.5)
    if null_val is not None:
        ax.axhline(null_val, color="black", linestyle="--", linewidth=1.8,
                   alpha=0.85, label=f"$C_{{null}}$ ({fmt_val(null_val, is_token)})")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(xg)
    ax.set_xticklabels(group_dims, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    if is_token:
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    else:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 107)


def make_group_figure(metric_specs, suptitle, outname):
    if not group_dims:
        return
    n_cols = len(metric_specs)
    if args.media:
        fig = plt.figure(figsize=(max(8, len(group_dims) * 2 + 6), 4.5))
        gs  = fig.add_gridspec(1, n_cols, wspace=0.3)
        for ci, (metric, title, ylabel, nidx, is_tok) in enumerate(metric_specs):
            ax = fig.add_subplot(gs[0, ci])
            null_val = mean_ignore_none(
                [(null_tokens if is_tok else null_scores)[cft][nidx] for cft in active_cfts])
            group_panel(ax, metric, f"{title} (avg over CFTs)", ylabel,
                        null_val, is_tok, cft=None)
    else:
        n_rows = len(active_cfts)
        fig = plt.figure(figsize=(max(8, len(group_dims) * 2 + 6), n_rows * 4.2))
        gs  = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.3)
        for ri, cft in enumerate(active_cfts):
            for ci, (metric, title, ylabel, nidx, is_tok) in enumerate(metric_specs):
                ax = fig.add_subplot(gs[ri, ci])
                null_val = (null_tokens if is_tok else null_scores)[cft][nidx]
                group_panel(ax, metric, f"{cft.upper()} {title}", ylabel,
                            null_val, is_tok, cft=cft)

    fig.suptitle(suptitle + (" [avg over CFTs]" if args.media else ""), fontsize=14)
    out = os.path.join(HERE, outname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out}")


make_group_figure(
    [
        ("req", "Satisfied Requirements", "% satisfied", 0, False),
        ("con", "Satisfied Constraints",  "% satisfied", 1, False),
    ],
    f"Group Averages -- KG injection req/con ({freeze_str})",
    f"plot_inj_avg_{freeze_tag}{cft_tag}{media_tag}.png",
)

make_group_figure(
    [
        ("prompt", "Input Tokens (prompt)",      "tokens", 0, True),
        ("compl",  "Output Tokens (completion)", "tokens", 1, True),
    ],
    f"Group Averages -- KG injection tokens ({freeze_str})",
    f"plot_inj_avg_tokens_{freeze_tag}{cft_tag}{media_tag}.png",
)

plt.show()
