import os
import re
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT        = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.join(ROOT, "experiments")

CFT_META = {
    "belval":  {"total_req": 23, "total_con": 19},
    "chrb":    {"total_req": 25, "total_con": 23},
    "cabinet": {"total_req": 27, "total_con": 26},
}

ALL_CONFIGS = [
   
    ("phi4",  "SCHEMA",    "JSON",    "TEXT",    "phi4/SCHEMA/JSON/TEXT"),
    ("phi4",  "SCHEMA",    "JSON",    "NO_TEXT", "phi4/SCHEMA/JSON/NO_TEXT"),
    ("phi4",  "SCHEMA",    "TURTLE",  "TEXT",    "phi4/SCHEMA/TURTLE/TEXT"),
    ("phi4",  "SCHEMA",    "TURTLE",  "NO_TEXT", "phi4/SCHEMA/TURTLE/NO_TEXT"),
    ("phi4",  "SCHEMA",    "YAML_LD", "TEXT",    "phi4/SCHEMA/YAML_LD/TEXT"),
    ("phi4",  "SCHEMA",    "YAML_LD", "NO_TEXT", "phi4/SCHEMA/YAML_LD/NO_TEXT"),
    ("phi4",  "NO_SCHEMA", "JSON",    "TEXT",    "phi4/NO_SCHEMA/JSON/TEXT"),
    ("phi4",  "NO_SCHEMA", "JSON",    "NO_TEXT", "phi4/NO_SCHEMA/JSON/NO_TEXT"),
    ("phi4",  "NO_SCHEMA", "TURTLE",  "TEXT",    "phi4/NO_SCHEMA/TURTLE/TEXT"),
    ("phi4",  "NO_SCHEMA", "TURTLE",  "NO_TEXT", "phi4/NO_SCHEMA/TURTLE/NO_TEXT"),
    ("phi4",  "NO_SCHEMA", "YAML_LD", "TEXT",    "phi4/NO_SCHEMA/YAML_LD/TEXT"),
    ("phi4",  "NO_SCHEMA", "YAML_LD", "NO_TEXT", "phi4/NO_SCHEMA/YAML_LD/NO_TEXT"),
    ("llama", None,        "JSON",    "TEXT",    "llama/JSON/TEXT"),
    ("llama", None,        "JSON",    "NO_TEXT", "llama/JSON/NO_TEXT"),
    ("llama", None,        "TURTLE",  "TEXT",    "llama/TURTLE/TEXT"),
    ("llama", None,        "TURTLE",  "NO_TEXT", "llama/TURTLE/NO_TEXT"),
    ("llama", None,        "YAML_LD", "TEXT",    "llama/YAML_LD/TEXT"),
    ("llama", None,        "YAML_LD", "NO_TEXT", "llama/YAML_LD/NO_TEXT"),
]

DIM_ORDER  = ["extractor", "schema", "format", "text"]
DIM_VALUES = {
    "extractor": ["phi4",   "llama"],
    "schema":    ["SCHEMA", "NO_SCHEMA"],
    "format":    ["JSON",   "TURTLE",  "YAML_LD"],
    "text":      ["TEXT",   "NO_TEXT"],
}
DIM_SHORT = {
    "extractor": {"phi4": "M_2",  "llama": "M_1"},
    "schema":    {"SCHEMA": "S", "NO_SCHEMA": "NS"},
    "format":    {"JSON":   "J", "TURTLE":    "TL", "YAML_LD": "YL"},
    "text":      {"TEXT":   "T", "NO_TEXT":   "NT"},
}
PALETTE = ["#1565C0", "#42A5F5", "#0D47A1", "#90CAF9",
           "#2E7D32", "#66BB6A", "#1B5E20", "#A5D6A7",
           "#E65100", "#FF8F00", "#BF360C", "#FFCA28",
           "#6A1B9A", "#AB47BC", "#4A148C", "#CE93D8",
           "#00695C", "#26A69A"]
CFT_COLORS = ["#1565C0", "#E65100", "#2E7D32", "#9C27B0"]

parser = argparse.ArgumentParser()
parser.add_argument("--cft",       choices=list(CFT_META.keys()))
parser.add_argument("--extractor", choices=DIM_VALUES["extractor"])
parser.add_argument("--schema",    choices=DIM_VALUES["schema"])
parser.add_argument("--format",    choices=DIM_VALUES["format"])
parser.add_argument("--text",      choices=DIM_VALUES["text"])
parser.add_argument("--media", action="store_true",
                    help="Average across all CFTs instead of one bar/row per CFT")
args = parser.parse_args()

freeze = {d: getattr(args, d) for d in DIM_ORDER if getattr(args, d) is not None}
extractor_free = "extractor" not in freeze

active_cfts = [args.cft] if args.cft else list(CFT_META.keys())


def read_scores(rel_path, cft):
    path = os.path.join(EXPERIMENTS, cft, rel_path,
                        "C_{OAP}_0", "single_validation_kgagents_tri.txt")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    req = len(re.findall(r'^REQ-\d+', content, re.MULTILINE))
    con = len(re.findall(r'^CON-\d+', content, re.MULTILINE))
    return req, con


def read_tokens(rel_path, cft):
    path = os.path.join(EXPERIMENTS, cft, rel_path,
                        "C_{OAP}_0", "tokens.json")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["prompt_tokens"], data["completion_tokens"]


def read_null_tokens(cft):
    path = os.path.join(EXPERIMENTS, cft, "C_null", "C_null_0", "tokens.json")
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


def cfg_val(cfg, dim):
    ext, sch, fmt, txt, _ = cfg
    return {"extractor": ext, "schema": sch, "format": fmt, "text": txt}[dim]


def matches_freeze(cfg, freeze):
    for d, v in freeze.items():
        val = cfg_val(cfg, d)
        if val is None:
            continue
        if val != v:
            return False
    return True


def apply_constraint(cfg):
    if extractor_free:
        ext, sch, _, _, _ = cfg
        if ext == "phi4" and sch == "NO_SCHEMA":
            return False
    return True


free = [d for d in DIM_ORDER if d not in freeze]
if extractor_free and "schema" in free:
    free.remove("schema")


def sort_key(cfg):
    return tuple(
        DIM_VALUES[d].index(cfg_val(cfg, d)) if cfg_val(cfg, d) is not None else 999
        for d in free
    )


filtered = sorted(
    [c for c in ALL_CONFIGS if matches_freeze(c, freeze) and apply_constraint(c)],
    key=sort_key,
)

if not filtered:
    raise SystemExit("No configurations match the selected filters.")

filtered_avg = sorted(
    [c for c in ALL_CONFIGS if matches_freeze(c, freeze)],
    key=sort_key,
)

labels = []
for cfg in filtered:
    parts = [DIM_SHORT[d][cfg_val(cfg, d)] for d in free if cfg_val(cfg, d) is not None]
    labels.append("$" + "/".join(parts) + "$" if parts else cfg[-1])

per_cft_req = {cft: [] for cft in active_cfts}
per_cft_con = {cft: [] for cft in active_cfts}
per_cft_prompt = {cft: [] for cft in active_cfts}
per_cft_completion = {cft: [] for cft in active_cfts}

for cfg in filtered:
    _, _, _, _, rel_path = cfg
    for cft in active_cfts:
        req, con = read_scores(rel_path, cft)
        per_cft_req[cft].append(req / CFT_META[cft]["total_req"] * 100 if req is not None else 0.0)
        per_cft_con[cft].append(con / CFT_META[cft]["total_con"] * 100 if con is not None else 0.0)
        prompt, completion = read_tokens(rel_path, cft)
        per_cft_prompt[cft].append(prompt if prompt is not None else 0)
        per_cft_completion[cft].append(completion if completion is not None else 0)


per_cft_req_avg  = {cft: [] for cft in active_cfts}
per_cft_con_avg  = {cft: [] for cft in active_cfts}
per_cft_prompt_avg     = {cft: [] for cft in active_cfts}
per_cft_completion_avg = {cft: [] for cft in active_cfts}

for cfg in filtered_avg:
    _, _, _, _, rel_path = cfg
    for cft in active_cfts:
        req, con = read_scores(rel_path, cft)
        per_cft_req_avg[cft].append(req / CFT_META[cft]["total_req"] * 100 if req is not None else 0.0)
        per_cft_con_avg[cft].append(con / CFT_META[cft]["total_con"] * 100 if con is not None else 0.0)
        prompt, completion = read_tokens(rel_path, cft)
        per_cft_prompt_avg[cft].append(prompt if prompt is not None else 0)
        per_cft_completion_avg[cft].append(completion if completion is not None else 0)


GROUPS = {}
for dim in free:
    opts = {}
    for val in DIM_VALUES[dim]:
        idxs = [i for i, cfg in enumerate(filtered) if cfg_val(cfg, dim) == val]
        if idxs:
            opts[val] = idxs
    if len(opts) >= 2:
        GROUPS[dim] = opts

GROUPS_AVG = {}
for dim in free:
    opts = {}
    if dim == "extractor" and extractor_free:
        phi4_s  = [i for i, c in enumerate(filtered_avg)
                   if cfg_val(c, "extractor") == "phi4" and cfg_val(c, "schema") == "SCHEMA"]
        phi4_ns = [i for i, c in enumerate(filtered_avg)
                   if cfg_val(c, "extractor") == "phi4" and cfg_val(c, "schema") == "NO_SCHEMA"]
        llama_i = [i for i, c in enumerate(filtered_avg)
                   if cfg_val(c, "extractor") == "llama"]
        if phi4_s:   opts["phi4/S"]  = phi4_s
        if phi4_ns:  opts["phi4/NS"] = phi4_ns
        if llama_i:  opts["llama"]   = llama_i
    else:
        for val in DIM_VALUES[dim]:
            idxs = [i for i, c in enumerate(filtered_avg) if cfg_val(c, dim) == val]
            if idxs:
                opts[val] = idxs
    if len(opts) >= 2:
        GROUPS_AVG[dim] = opts

GROUPS_AVG.pop("text", None)
if "text" in freeze:
    text_sides = [freeze["text"]]
else:
    text_sides = [tv for tv in DIM_VALUES["text"]
                  if any(cfg_val(c, "text") == tv for c in filtered_avg)]
n_text = len(text_sides)
text_idx_sets = {tv: {i for i, c in enumerate(filtered_avg) if cfg_val(c, "text") == tv}
                 for tv in text_sides}

n_groups     = len(GROUPS)
n_groups_avg = len(GROUPS_AVG)
n_cfts     = len(active_cfts)
cft_label  = args.cft.upper() if args.cft else " + ".join(c.upper() for c in active_cfts)
freeze_str = ", ".join(f"{d}={v}" for d, v in freeze.items()) if freeze else "no freeze"
cft_tag    = f"_{args.cft}" if args.cft else ""
freeze_tag = "_".join(freeze.values()) if freeze else "all"

null_scores = {cft: read_null_scores(cft) for cft in active_cfts}
null_tokens = {cft: read_null_tokens(cft) for cft in active_cfts}


def mean_ignore_none(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


media_tag       = "_media" if args.media else ""
media_null_req  = mean_ignore_none([null_scores[cft][0] for cft in active_cfts])
media_null_con  = mean_ignore_none([null_scores[cft][1] for cft in active_cfts])
media_null_prompt     = mean_ignore_none([null_tokens[cft][0] for cft in active_cfts])
media_null_completion = mean_ignore_none([null_tokens[cft][1] for cft in active_cfts])

fig1 = plt.figure(figsize=(max(8, len(filtered) * 1.5), 9))
gs1  = fig1.add_gridspec(2, 1, hspace=0.45)
ax_req = fig1.add_subplot(gs1[0])
ax_con = fig1.add_subplot(gs1[1])

x = np.arange(len(labels))

if args.media:
    bar_width = 0.5
    for ax, data_by_cft, title, bar_color, null_mean in [
        (ax_req, per_cft_req, "Satisfied Requirements", CFT_COLORS[0], media_null_req),
        (ax_con, per_cft_con, "Satisfied Constraints",  CFT_COLORS[1], media_null_con),
    ]:
        means, stds = [], []
        for i in range(len(filtered)):
            vals = [data_by_cft[cft][i] for cft in active_cfts]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        bars = ax.bar(x, means, width=bar_width, color=bar_color, edgecolor="white")
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 1.5,
                    f"{mean:.0f}%\n±{std:.0f}%", ha="center", va="bottom", fontsize=7)
        if null_mean is not None:
            ax.axhline(null_mean, color="black", linestyle="--", linewidth=1.8,
                       alpha=0.85, label=f"$C_{{null}}$ ({null_mean:.0f}%)")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("% satisfied", fontsize=11)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 107)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="upper right")
else:
    bar_width = 0.6 / n_cfts
    for ax, data_by_cft, title, null_idx in [
        (ax_req, per_cft_req, "Satisfied Requirements", 0),
        (ax_con, per_cft_con, "Satisfied Constraints",  1),
    ]:
        for ci, cft in enumerate(active_cfts):
            offset = (ci - (n_cfts - 1) / 2) * bar_width
            xpos   = x + offset
            bars   = ax.bar(xpos, data_by_cft[cft], width=bar_width * 0.9,
                            color=CFT_COLORS[ci % len(CFT_COLORS)], edgecolor="white",
                            label=cft.upper())
            for bar, pct in zip(bars, data_by_cft[cft]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=7)
            nv = null_scores[cft][null_idx]
            if nv is not None:
                ax.axhline(nv, color=CFT_COLORS[ci % len(CFT_COLORS)], linestyle="--",
                           linewidth=1.8, alpha=0.85,
                           label=f"{cft.upper()} $C_{{null}}$")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("% satisfied", fontsize=11)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, 107)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles, labels_leg = ax_req.get_legend_handles_labels()
    fig1.legend(handles, labels_leg, fontsize=9, ncol=len(handles),
                loc="lower center", bbox_to_anchor=(0.5, -0.04))

fig1.suptitle(f"{cft_label}: $C_{{OAP}}$, $\\gamma=0$ ({freeze_str})"
              + (" [avg over CFTs]" if args.media else ""), fontsize=14)
out1 = os.path.join(ROOT, f"plot_{freeze_tag}{cft_tag}{media_tag}.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {out1}")

if n_groups_avg > 0:
    series_colors = ["#5C6BC0", "#EF5350", "#43A047", "#FF8F00",
                     "#8E24AA", "#00ACC1", "#D81B60", "#7CB342"]
    group_dims_avg = list(GROUPS_AVG.keys())
    series_avg   = [(dim, key) for dim in group_dims_avg for key in GROUPS_AVG[dim].keys()]
    n_series_avg = len(series_avg)
    xt = np.arange(n_text)

    if args.media:
        per_config_req_avg = [float(np.mean([per_cft_req_avg[cft][i] for cft in active_cfts]))
                               for i in range(len(filtered_avg))]
        per_config_con_avg = [float(np.mean([per_cft_con_avg[cft][i] for cft in active_cfts]))
                               for i in range(len(filtered_avg))]

        fig2 = plt.figure(figsize=(max(8, n_series_avg * n_text * 1.1 + 4), 4))
        gs2  = fig2.add_gridspec(1, 2, wspace=0.35)

        for ci, (per_config_pcts, metric, null_mean) in enumerate([
            (per_config_req_avg, "Requirements", media_null_req),
            (per_config_con_avg, "Constraints",  media_null_con),
        ]):
            ax = fig2.add_subplot(gs2[0, ci])
            w = 0.8 / n_series_avg
            for si, (dim, key) in enumerate(series_avg):
                offset = (si - (n_series_avg - 1) / 2) * w
                for ti, tv in enumerate(text_sides):
                    vals = [per_config_pcts[i] for i in GROUPS_AVG[dim][key] if i in text_idx_sets[tv]]
                    if not vals:
                        continue
                    mean = np.mean(vals)
                    ax.bar(xt[ti] + offset, mean, w,
                           color=series_colors[si % len(series_colors)], edgecolor="white",
                           label=key if ti == 0 else None)
                    ax.text(xt[ti] + offset, mean + 1, f"{key}\n{mean:.0f}%",
                            ha="center", va="bottom", fontsize=7.5)
            if null_mean is not None:
                ax.axhline(null_mean, color="black", linestyle="--", linewidth=1.8,
                           alpha=0.85, label=f"$C_{{null}}$")
            ax.set_title(f"{metric} (avg over CFTs)", fontsize=11)
            ax.set_xticks(xt)
            ax.set_xticklabels(text_sides, fontsize=10)
            ax.set_ylabel("% satisfied", fontsize=10)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_ylim(0, 107)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8, loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    else:
        fig2 = plt.figure(figsize=(max(8, n_series_avg * n_text * 1.1 + 4), n_cfts * 4))
        gs2  = fig2.add_gridspec(n_cfts, 2, hspace=0.5, wspace=0.35)

        for ri, cft in enumerate(active_cfts):
            for ci, (per_cft_pcts, metric) in enumerate([
                (per_cft_req_avg, "Requirements"),
                (per_cft_con_avg, "Constraints"),
            ]):
                ax = fig2.add_subplot(gs2[ri, ci])
                w = 0.8 / n_series_avg
                for si, (dim, key) in enumerate(series_avg):
                    offset = (si - (n_series_avg - 1) / 2) * w
                    for ti, tv in enumerate(text_sides):
                        vals = [per_cft_pcts[cft][i] for i in GROUPS_AVG[dim][key] if i in text_idx_sets[tv]]
                        if not vals:
                            continue
                        mean = np.mean(vals)
                        ax.bar(xt[ti] + offset, mean, w,
                               color=series_colors[si % len(series_colors)], edgecolor="white",
                               label=key if ti == 0 else None)
                        ax.text(xt[ti] + offset, mean + 1, f"{key}\n{mean:.0f}%",
                                ha="center", va="bottom", fontsize=7.5)
                nv = null_scores[cft][ci]
                if nv is not None:
                    ax.axhline(nv, color=CFT_COLORS[ri % len(CFT_COLORS)],
                               linestyle="--", linewidth=1.8, alpha=0.85,
                               label=f"$C_{{null}}$")
                ax.set_title(f"{cft.upper()} {metric}", fontsize=11)
                ax.set_xticks(xt)
                ax.set_xticklabels(text_sides, fontsize=10)
                ax.set_ylabel("% satisfied", fontsize=10)
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_ylim(0, 107)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                ax.legend(fontsize=8, loc="upper left",
                          bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    fig2.suptitle(f"Group Averages $C_{{OAP}}$, $\\gamma=0$ ({freeze_str})"
                  + (" [avg over CFTs]" if args.media else ""), fontsize=13)
    out2 = os.path.join(ROOT, f"plot_avg_{freeze_tag}{cft_tag}{media_tag}.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out2}")

plt.show()

fig3 = plt.figure(figsize=(max(8, len(filtered) * 1.5), 9))
gs3  = fig3.add_gridspec(2, 1, hspace=0.45)
ax_prompt = fig3.add_subplot(gs3[0])
ax_compl  = fig3.add_subplot(gs3[1])

if args.media:
    for ax, data_by_cft, title, bar_color, null_mean in [
        (ax_prompt, per_cft_prompt,     "Input Tokens (prompt)",      CFT_COLORS[0], media_null_prompt),
        (ax_compl,  per_cft_completion, "Output Tokens (completion)", CFT_COLORS[1], media_null_completion),
    ]:
        means, stds = [], []
        for i in range(len(filtered)):
            vals = [data_by_cft[cft][i] for cft in active_cfts]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        bars = ax.bar(x, means, width=bar_width, color=bar_color, edgecolor="white")
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, mean * 1.01,
                        f"{mean/1000:.0f}k\n±{std/1000:.0f}k", ha="center", va="bottom", fontsize=7)
        if null_mean is not None:
            ax.axhline(null_mean, color="black", linestyle="--", linewidth=1.8,
                       alpha=0.85, label=f"$C_{{null}}$ ({null_mean/1000:.0f}k)")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("tokens", fontsize=11)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="upper right")
else:
    for ax, data_by_cft, title, tok_idx in [
        (ax_prompt, per_cft_prompt,     "Input Tokens (prompt)",     0),
        (ax_compl,  per_cft_completion, "Output Tokens (completion)", 1),
    ]:
        for ci, cft in enumerate(active_cfts):
            offset = (ci - (n_cfts - 1) / 2) * bar_width
            xpos   = x + offset
            bars   = ax.bar(xpos, data_by_cft[cft], width=bar_width * 0.9,
                            color=CFT_COLORS[ci % len(CFT_COLORS)], edgecolor="white",
                            label=cft.upper())
            for bar, val in zip(bars, data_by_cft[cft]):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                            f"{val/1000:.0f}k", ha="center", va="bottom", fontsize=7)
            nv = null_tokens[cft][tok_idx]
            if nv is not None:
                ax.axhline(nv, color=CFT_COLORS[ci % len(CFT_COLORS)], linestyle="--",
                           linewidth=1.8, alpha=0.85,
                           label=f"{cft.upper()} $C_{{null}}$")
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("tokens", fontsize=11)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles3, labels3 = ax_prompt.get_legend_handles_labels()
    fig3.legend(handles3, labels3, fontsize=9, ncol=len(handles3),
                loc="lower center", bbox_to_anchor=(0.5, -0.04))

fig3.suptitle(f"{cft_label}: Token Usage $C_{{OAP}}$, $\\gamma=0$ ({freeze_str})"
              + (" [avg over CFTs]" if args.media else ""), fontsize=14)
out3 = os.path.join(ROOT, f"plot_tokens_{freeze_tag}{cft_tag}{media_tag}.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {out3}")

if n_groups_avg > 0:
    if args.media:
        per_config_prompt_avg     = [float(np.mean([per_cft_prompt_avg[cft][i] for cft in active_cfts]))
                                      for i in range(len(filtered_avg))]
        per_config_completion_avg = [float(np.mean([per_cft_completion_avg[cft][i] for cft in active_cfts]))
                                      for i in range(len(filtered_avg))]

        fig4 = plt.figure(figsize=(max(8, n_series_avg * n_text * 1.1 + 4), 4))
        gs4  = fig4.add_gridspec(1, 2, wspace=0.35)

        for ci, (per_config_toks, metric, null_mean) in enumerate([
            (per_config_prompt_avg,     "Input Tokens",  media_null_prompt),
            (per_config_completion_avg, "Output Tokens", media_null_completion),
        ]):
            ax = fig4.add_subplot(gs4[0, ci])
            w = 0.8 / n_series_avg
            for si, (dim, key) in enumerate(series_avg):
                offset = (si - (n_series_avg - 1) / 2) * w
                for ti, tv in enumerate(text_sides):
                    vals = [per_config_toks[i] for i in GROUPS_AVG[dim][key] if i in text_idx_sets[tv]]
                    if not vals:
                        continue
                    mean = np.mean(vals)
                    ax.bar(xt[ti] + offset, mean, w,
                           color=series_colors[si % len(series_colors)], edgecolor="white",
                           label=key if ti == 0 else None)
                    ax.text(xt[ti] + offset, mean * 1.01, f"{key}\n{mean/1000:.0f}k",
                            ha="center", va="bottom", fontsize=7.5)
            if null_mean is not None:
                ax.axhline(null_mean, color="black", linestyle="--", linewidth=1.8,
                           alpha=0.85, label=f"$C_{{null}}$")
            ax.set_title(f"{metric} (avg over CFTs)", fontsize=11)
            ax.set_xticks(xt)
            ax.set_xticklabels(text_sides, fontsize=10)
            ax.set_ylabel("tokens", fontsize=10)
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=8, loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    else:
        fig4 = plt.figure(figsize=(max(8, n_series_avg * n_text * 1.1 + 4), n_cfts * 4))
        gs4  = fig4.add_gridspec(n_cfts, 2, hspace=0.5, wspace=0.35)

        for ri, cft in enumerate(active_cfts):
            for ci, (per_cft_toks, metric) in enumerate([
                (per_cft_prompt_avg,     "Input Tokens"),
                (per_cft_completion_avg, "Output Tokens"),
            ]):
                ax = fig4.add_subplot(gs4[ri, ci])
                w = 0.8 / n_series_avg
                for si, (dim, key) in enumerate(series_avg):
                    offset = (si - (n_series_avg - 1) / 2) * w
                    for ti, tv in enumerate(text_sides):
                        vals = [per_cft_toks[cft][i] for i in GROUPS_AVG[dim][key] if i in text_idx_sets[tv]]
                        if not vals:
                            continue
                        mean = np.mean(vals)
                        ax.bar(xt[ti] + offset, mean, w,
                               color=series_colors[si % len(series_colors)], edgecolor="white",
                               label=key if ti == 0 else None)
                        ax.text(xt[ti] + offset, mean * 1.01, f"{key}\n{mean/1000:.0f}k",
                                ha="center", va="bottom", fontsize=7.5)
                nv = null_tokens[cft][ci]
                if nv is not None:
                    ax.axhline(nv, color=CFT_COLORS[ri % len(CFT_COLORS)],
                               linestyle="--", linewidth=1.8, alpha=0.85,
                               label=f"$C_{{null}}$")
                ax.set_title(f"{cft.upper()} {metric}", fontsize=11)
                ax.set_xticks(xt)
                ax.set_xticklabels(text_sides, fontsize=10)
                ax.set_ylabel("tokens", fontsize=10)
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                ax.legend(fontsize=8, loc="upper left",
                          bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    fig4.suptitle(f"Group Averages Token Usage $C_{{OAP}}$, $\\gamma=0$ ({freeze_str})"
                  + (" [avg over CFTs]" if args.media else ""), fontsize=13)
    out4 = os.path.join(ROOT, f"plot_avg_tokens_{freeze_tag}{cft_tag}{media_tag}.png")
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out4}")
