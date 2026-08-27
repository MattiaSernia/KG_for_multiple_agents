"""
Rank the 4 KG-injection points against each other, using only the llama
TEXT setting (TURTLE format, the only one with injection experiments).

Method
------
* The "configurations" being ranked are the 4 injection points:
      C_{O}  <  C_{OA} / C_{OP}  <  C_{OAP}
* For every "category" = (CFT x metric) -- 6 in total:
      belval-req, belval-con, chrb-req, chrb-con, cabinet-req, cabinet-con
  the 4 injection points are ranked by % of satisfied requirements /
  constraints: the best gets 1 point, the second 2, and so on.
* The per-category points are summed for each injection point.
  Lowest total = best configuration.

Ties inside a category share the average of the positions they occupy
(e.g. two configs tied for the top get (1+2)/2 = 1.5 each), so the total
number of points handed out per category is always 1+2+3+4 = 10.

Usage
-----
    python3 rank_injection.py            # full breakdown + final ranking
    python3 rank_injection.py --quiet    # only the final ranking table
"""
import os
import re
import argparse

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = os.path.join(ROOT, "experiments")

CFT_META = {
    "belval":  {"total_req": 23, "total_con": 19},
    "chrb":    {"total_req": 25, "total_con": 23},
    "cabinet": {"total_req": 27, "total_con": 26},
}
CFTS = list(CFT_META.keys())

MODES = ["C_{O}", "C_{OA}", "C_{OP}", "C_{OAP}"]
MODE_VALFILE = {
    "C_{O}":   "single_validation_kg.txt",
    "C_{OA}":  "single_validation_kgagents.txt",
    "C_{OP}":  "single_validation_kgcft.txt",
    "C_{OAP}": "single_validation_kgagents_tri.txt",
}


REL_PATH = "llama/TURTLE/TEXT"

METRICS = ["req", "con"]
METRIC_LABEL = {"req": "requirements", "con": "constraints"}


def read_pct(mode, cft, metric):
    """% of satisfied requirements/constraints for one injection point, or None."""
    path = os.path.join(EXPERIMENTS, cft, REL_PATH, f"{mode}_0", MODE_VALFILE[mode])
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    tag = "REQ" if metric == "req" else "CON"
    hits = len(re.findall(rf'^{tag}-\d+', content, re.MULTILINE))
    total = CFT_META[cft]["total_req" if metric == "req" else "total_con"]
    return hits / total * 100


def rank_points(pcts):
    """Given {mode: pct} return {mode: points}.

    Best (highest pct) -> 1 point, next -> 2, ... Missing data (None) is
    treated as the worst possible score (0%). Ties share the average of the
    positions they occupy.
    """
    scored = {m: (pcts.get(m) if pcts.get(m) is not None else 0.0) for m in MODES}
    order = sorted(MODES, key=lambda m: scored[m], reverse=True)
    points = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scored[order[j + 1]] == scored[order[i]]:
            j += 1
        avg = sum(range(i + 1, j + 2)) / (j - i + 1)
        for k in range(i, j + 1):
            points[order[k]] = avg
        i = j + 1
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the final ranking table")
    args = parser.parse_args()

    totals = {m: 0.0 for m in MODES}

    if not args.quiet:
        print(f"\n{'=' * 70}")
        print(f"SETTING: llama   ({REL_PATH})")
        print(f"{'=' * 70}")

    for cft in CFTS:
        for metric in METRICS:
            pcts = {m: read_pct(m, cft, metric) for m in MODES}
            points = rank_points(pcts)
            for m in MODES:
                totals[m] += points[m]

            if not args.quiet:
                print(f"\n  {cft}-{METRIC_LABEL[metric]}")
                for m in sorted(MODES, key=lambda m: points[m]):
                    pv = pcts[m]
                    pv_s = f"{pv:5.1f}%" if pv is not None else "  n/a "
                    print(f"    {m:8s}  {pv_s}   -> {points[m]:.1f} pt")

    ranking = sorted(MODES, key=lambda m: totals[m])
    print(f"\n  FINAL RANKING (llama)  [lowest = best]")
    print(f"  {'rank':<5}{'injection':<10}{'total points':<14}")
    for pos, m in enumerate(ranking, 1):
        print(f"  {pos:<5}{m:<10}{totals[m]:<14.1f}")
    print()


if __name__ == "__main__":
    main()
