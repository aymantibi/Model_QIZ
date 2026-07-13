#!/usr/bin/env python3
"""
estimate_beta_s.py
------------------
Estimates sector demand weights beta_s from the Egypt 2008 Input-Output Table.

Method (from appendix):
    Absorption_s = Output_s + Imports_s - Exports_s
    beta_s = Absorption_s / sum_s' Absorption_s'

Two sector groupings are computed:

  GROUPING 1 — Baseline (2 sectors):
    T  : Textiles + Wearing apparel (ISIC 13, 14)
    O  : All other manufacturing (ISIC 10-12, 15-33)

  GROUPING 2 — Extended (5 sectors):
    T  : Textiles + Wearing apparel (ISIC 13, 14)
    S1 : Food products (ISIC 10)
    S2 : Chemicals (ISIC 20-21)
    S3 : Non-metallic minerals (ISIC 23)
    O  : All remaining manufacturing

IO table sheet: "IO _2"
Columns used (0-indexed):
    col58: total exports (goods + services)
    col60: total imports CIF (stored as positive)
    col62: total output
Units: millions of Egyptian pounds, 2008.
"""

import json
import os
import xlrd

IO_PATH = r"C:\Users\Admin\Desktop\Idea QIZs and Development\IO Tables\Egypt 2008 IO Tables.xls"
OUT_PATH = os.path.join(os.path.dirname(__file__), "params_estimated.json")

# ── IO table row indices (0-based) ───────────────────────────────────────────
# Row 14=ISIC10, 15=ISIC11, 16=ISIC12, 17=ISIC13, 18=ISIC14, 19=ISIC15,
# 20=ISIC16, 21=ISIC17-18, 22=ISIC19, 23=ISIC20-21, 24=ISIC22, 25=ISIC23,
# 26=ISIC24, 27=ISIC25, 28=ISIC26, 29=ISIC27, 30=ISIC28, 31=ISIC29-30,
# 32=ISIC31-33

ROWS_T  = [17, 18]   # ISIC 13 (textiles) + 14 (wearing apparel)
ROWS_S1 = [14]       # ISIC 10  food products
ROWS_S2 = [23]       # ISIC 20-21  chemicals
ROWS_S3 = [25]       # ISIC 23  non-metallic minerals

ALL_MFG_ROWS = list(range(14, 33))

# Other manufacturing = all mfg minus T and the three named sectors
ROWS_O_BASE     = [r for r in ALL_MFG_ROWS if r not in ROWS_T]
ROWS_O_EXTENDED = [r for r in ALL_MFG_ROWS
                   if r not in ROWS_T + ROWS_S1 + ROWS_S2 + ROWS_S3]

# Column indices (0-based)
COL_EXPORTS = 58
COL_IMPORTS = 60
COL_OUTPUT  = 62


# ── helpers ──────────────────────────────────────────────────────────────────

def load_io_sheet(path: str):
    wb = xlrd.open_workbook(path)
    return wb.sheet_by_name("IO _2")


def sector_absorption(sh, rows: list) -> dict:
    output  = sum(sh.cell_value(r, COL_OUTPUT)  for r in rows)
    exports = sum(sh.cell_value(r, COL_EXPORTS) for r in rows)
    imports = sum(sh.cell_value(r, COL_IMPORTS) for r in rows)
    return {
        "output":     round(output,  1),
        "imports":    round(imports, 1),
        "exports":    round(exports, 1),
        "absorption": round(output + imports - exports, 1),
    }


def compute_betas(stats: dict) -> dict:
    """Given {sector: stats_dict}, return {sector: beta}."""
    total = sum(v["absorption"] for v in stats.values())
    return {s: round(v["absorption"] / total, 6) for s, v in stats.items()}


# ── main estimation ──────────────────────────────────────────────────────────

def estimate_beta(path: str = IO_PATH) -> dict:
    sh = load_io_sheet(path)

    # ── Grouping 1: T + O ────────────────────────────────────────────────────
    g1_stats = {
        "T": sector_absorption(sh, ROWS_T),
        "O": sector_absorption(sh, ROWS_O_BASE),
    }
    g1_beta = compute_betas(g1_stats)

    # ── Grouping 2: T + S1 + S2 + S3 + S4 + O ───────────────────────────────
    g2_stats = {
        "T":  sector_absorption(sh, ROWS_T),
        "S1": sector_absorption(sh, ROWS_S1),
        "S2": sector_absorption(sh, ROWS_S2),
        "S3": sector_absorption(sh, ROWS_S3),
        "O":  sector_absorption(sh, ROWS_O_EXTENDED),
    }
    g2_beta = compute_betas(g2_stats)

    sector_labels = {
        "T":  "Textiles + Wearing apparel (ISIC 13-14)",
        "S1": "Food products (ISIC 10)",
        "S2": "Chemicals (ISIC 20-21)",
        "S3": "Non-metallic minerals (ISIC 23)",
        "O":  "Other manufacturing (residual)",
    }

    return {
        "grouping_1": {
            "description": "Baseline: T (textiles+apparel) + O (other mfg)",
            "sectors": list(g1_stats.keys()),
            "beta":        g1_beta,
            "diagnostics": g1_stats,
        },
        "grouping_2": {
            "description": "Extended: T + top-4 sectors by absorption + O (residual)",
            "sectors": list(g2_stats.keys()),
            "beta":        g2_beta,
            "diagnostics": g2_stats,
        },
        "sector_labels": sector_labels,
        "source": "Egypt IO Table 2008, sheet 'IO _2'",
        "method": "beta_s = Absorption_s / sum_s Absorption_s; "
                  "Absorption = Output + Imports - Exports (col58, col60, col62)",
        "units": "millions of Egyptian pounds, 2008",
    }


def update_params_file(result: dict, out_path: str = OUT_PATH):
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            params = json.load(f)
    else:
        params = {}
    params["beta_s"] = result
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    return out_path


# ── reporting ────────────────────────────────────────────────────────────────

def print_grouping(name: str, g: dict, labels: dict):
    sep = "-" * 60
    print(f"\n  {sep}")
    print(f"  {name}: {g['description']}")
    print(f"  {sep}")
    print(f"  {'Sector':<6}  {'Label':<42}  {'Absorption':>12}  {'beta':>8}")
    print(f"  {sep}")
    total_abs = sum(v["absorption"] for v in g["diagnostics"].values())
    for s in g["sectors"]:
        d = g["diagnostics"][s]
        lbl = labels.get(s, s)
        print(f"  {s:<6}  {lbl:<42}  {d['absorption']:>12,.1f}  {g['beta'][s]:>8.6f}")
    print(f"  {sep}")
    print(f"  {'Total':<50}  {total_abs:>12,.1f}  {sum(g['beta'].values()):>8.6f}")


if __name__ == "__main__":
    result = estimate_beta()

    print("=" * 70)
    print("  beta_s: sector demand weights (manufacturing absorption shares)")
    print("=" * 70)

    for gname, gkey in [("Grouping 1 — Baseline", "grouping_1"),
                        ("Grouping 2 — Extended", "grouping_2")]:
        print_grouping(gname, result[gkey], result["sector_labels"])

    saved = update_params_file(result)
    print(f"\n  Saved to: {saved}")
