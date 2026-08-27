import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

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

REQ_COLOR  = "#1565C0"   
CON_COLOR  = "#E65100"   
GROUP_COLORS = ["#5C6BC0", "#EF5350", "#43A047", "#FF8F00"]

parser = argparse.ArgumentParser()
parser.add_argument("--by", choices=["format", "extractor"], default="format",
                    help="Group bars by KG format (default) or by extractor")
args = parser.parse_args()


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


null_req_vals, null_con_vals = [], []
for cft in CFTS:
    nr, nc = read_null_scores(cft)
    if nr is not None:
        null_req_vals.append(nr)
    if nc is not None:
        null_con_vals.append(nc)

null_req_mean = np.mean(null_req_vals) if null_req_vals else None
null_con_mean = np.mean(null_con_vals) if null_con_vals else None

if args.by == "format":
    bucket_req = {fmt: [] for fmt in FORMATS}
    bucket_con = {fmt: [] for fmt in FORMATS}

    for ext, sch, fmt, txt, rel_path in ALL_CONFIGS:
        for cft in CFTS:
            req, con = read_scores(rel_path, cft)
            if req is not None:
                bucket_req[fmt].append(req / CFT_META[cft]["total_req"] * 100)
            if con is not None:
                bucket_con[fmt].append(con / CFT_META[cft]["total_con"] * 100)

    keys       = FORMATS
    key_labels = [FORMAT_LABEL[f] for f in FORMATS]
    out_name   = "plot_format_report_format.png"
    title      = ("Mean Satisfied Requirements & Constraints by KG Format\n"
                  "(averaged across all models, schemas, text variants and CFTs)")

else:  
    bucket_req = {gname: [] for gname, _ in EXTRACTOR_GROUPS}
    bucket_con = {gname: [] for gname, _ in EXTRACTOR_GROUPS}

    for ext, sch, fmt, txt, rel_path in ALL_CONFIGS:
        for gname, pred in EXTRACTOR_GROUPS:
            if pred(ext, sch):
                for cft in CFTS:
                    req, con = read_scores(rel_path, cft)
                    if req is not None:
                        bucket_req[gname].append(req / CFT_META[cft]["total_req"] * 100)
                    if con is not None:
                        bucket_con[gname].append(con / CFT_META[cft]["total_con"] * 100)
                break

    keys       = [gname for gname, _ in EXTRACTOR_GROUPS]
    key_labels = keys
    out_name   = "plot_format_report_extractor.png"
    title      = ("Mean Satisfied Requirements & Constraints by Extractor\n"
                  "(averaged across formats, text variants and CFTs)")

x         = np.arange(len(keys))
bar_width = 0.32

fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.5), 5))

for i, key in enumerate(keys):
    req_mean = np.mean(bucket_req[key]) if bucket_req[key] else 0.0
    con_mean = np.mean(bucket_con[key]) if bucket_con[key] else 0.0

    bar_r = ax.bar(x[i] - bar_width / 2, req_mean, bar_width,
                   color=REQ_COLOR, edgecolor="white")
    bar_c = ax.bar(x[i] + bar_width / 2, con_mean, bar_width,
                   color=CON_COLOR, edgecolor="white")

    ax.text(bar_r[0].get_x() + bar_r[0].get_width() / 2, req_mean + 1.5,
            f"{req_mean:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.text(bar_c[0].get_x() + bar_c[0].get_width() / 2, con_mean + 1.5,
            f"{con_mean:.1f}%", ha="center", va="bottom", fontsize=7)

if null_req_mean is not None:
    ax.axhline(null_req_mean, color=REQ_COLOR, linestyle="--", linewidth=1.8,
               alpha=0.85, label=f"$C_{{\emptyset}}$ Requirements ({null_req_mean:.1f}%)")
if null_con_mean is not None:
    ax.axhline(null_con_mean, color=CON_COLOR, linestyle="--", linewidth=1.8,
               alpha=0.85, label=f"$C_{{\emptyset}}$ Constraints ({null_con_mean:.1f}%)")

ax.set_xticks(x)
ax.set_xticklabels(key_labels, fontsize=10)
ax.set_ylabel("% satisfied", fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 107)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_title(title, fontsize=13)

req_patch = mpatches.Patch(color=REQ_COLOR, label="Requirements (mean %)")
con_patch = mpatches.Patch(color=CON_COLOR, label="Constraints (mean %)")
handles, leg_labels = ax.get_legend_handles_labels()
fig.legend(handles=[req_patch, con_patch] + handles,
           labels=["Requirements (mean %)", "Constraints (mean %)"] + leg_labels,
           fontsize=9, ncol=2 + len(handles),
           loc="lower center", bbox_to_anchor=(0.5, -0.06))

fig.tight_layout()
out = os.path.join(SCRIPT_DIR, out_name)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
