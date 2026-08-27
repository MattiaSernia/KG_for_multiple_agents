import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(SCRIPT_DIR)
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

CFTS = list(CFT_META.keys())

FORMATS      = ["JSON", "TURTLE", "YAML_LD"]
FORMAT_LABEL = {"JSON": "JSON", "TURTLE": "Turtle", "YAML_LD": "YAML-LD"}

EXTRACTOR_GROUPS = [
    (r"$M_2$/S",  lambda e, s: e == "phi4" and s == "SCHEMA"),
    (r"$M_2$/NS", lambda e, s: e == "phi4" and s == "NO_SCHEMA"),
    (r"$M_1$",   lambda e, s: e == "llama"),
]

CFT_COLORS   = ["#1565C0", "#E65100", "#2E7D32", "#9C27B0"]
GROUP_COLORS = ["#5C6BC0", "#EF5350", "#43A047", "#FF8F00"]

parser = argparse.ArgumentParser()
parser.add_argument("--by", choices=["format", "extractor"], default="format",
                    help="Group bars by KG format (default) or by extractor")
args = parser.parse_args()


def read_tokens(rel_path, cft):
    path = os.path.join(EXPERIMENTS, cft, rel_path,
                        "C_{OAP}_0", "tokens.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["prompt_tokens"]


def read_null_tokens(cft):
    path = os.path.join(EXPERIMENTS, cft, "C_null", "C_null_0", "tokens.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["prompt_tokens"]


null_vals = [read_null_tokens(cft) for cft in CFTS]
null_vals = [v for v in null_vals if v is not None]
null_mean = np.mean(null_vals) if null_vals else None

fig, ax = plt.subplots(figsize=(8, 5))

if args.by == "format":
    bucket = {fmt: [] for fmt in FORMATS}
    for ext, sch, fmt, txt, rel_path in ALL_CONFIGS:
        for cft in CFTS:
            pt = read_tokens(rel_path, cft)
            if pt is not None:
                bucket[fmt].append(pt)

    x         = np.arange(len(FORMATS))
    bar_width = 0.5
    for i, fmt in enumerate(FORMATS):
        mean = np.mean(bucket[fmt]) if bucket[fmt] else 0.0
        bar  = ax.bar(x[i], mean, bar_width,
                      color=CFT_COLORS[i % len(CFT_COLORS)], edgecolor="white")
        if mean > 0:
            ax.text(bar[0].get_x() + bar[0].get_width() / 2, mean * 1.01,
                    f"{mean/1000:.0f}k", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([FORMAT_LABEL[f] for f in FORMATS], fontsize=10)
    ax.set_title("Mean Input Token Usage by KG Format\n"
                 "(averaged across all models, schemas, text variants and CFTs)", fontsize=13)
    out_name = "plot_tokens_report_format.png"

else:  
    bucket = {gname: [] for gname, _ in EXTRACTOR_GROUPS}
    for ext, sch, fmt, txt, rel_path in ALL_CONFIGS:
        for gname, pred in EXTRACTOR_GROUPS:
            if pred(ext, sch):
                for cft in CFTS:
                    pt = read_tokens(rel_path, cft)
                    if pt is not None:
                        bucket[gname].append(pt)
                break

    group_names = [gname for gname, _ in EXTRACTOR_GROUPS]
    x           = np.arange(len(group_names))
    bar_width   = 0.5
    for i, gname in enumerate(group_names):
        mean = np.mean(bucket[gname]) if bucket[gname] else 0.0
        bar  = ax.bar(x[i], mean, bar_width,
                      color=GROUP_COLORS[i % len(GROUP_COLORS)], edgecolor="white")
        if mean > 0:
            ax.text(bar[0].get_x() + bar[0].get_width() / 2, mean * 1.01,
                    f"{mean/1000:.0f}k", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_title("Mean Input Token Usage by Extractor\n"
                 "(averaged across formats, text variants and CFTs)", fontsize=13)
    out_name = "plot_tokens_report_extractor.png"

if null_mean is not None:
    ax.axhline(null_mean, color="black", linestyle="--", linewidth=1.8,
               alpha=0.85, label=f"$C_{{\emptyset}}$ ({null_mean/1000:.0f}k)")

ax.set_ylabel("Input tokens", fontsize=11)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.grid(axis="y", linestyle="--", alpha=0.4)

handles, leg_labels = ax.get_legend_handles_labels()
fig.legend(handles, leg_labels, fontsize=9, ncol=len(handles),
           loc="lower center", bbox_to_anchor=(0.5, -0.06))

fig.tight_layout()
out = os.path.join(SCRIPT_DIR, out_name)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
