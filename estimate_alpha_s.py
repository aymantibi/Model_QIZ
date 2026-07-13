#!/usr/bin/env python3
"""
estimate_alpha_s.py
-------------------
Estimates labor share in production alpha_s from the Egypt 2008 Input-Output Table.

Method (from appendix):
    alpha_s = LaborCompensation_s / (LaborCompensation_s + IntermediateInputs_s)

This is the labor cost share in variable production costs, consistent with the
Cobb-Douglas production function y = phi * l^alpha * m^(1-alpha).

IO table sheet: "IO _2"
Rows used (0-indexed):
    row62: wages / labor compensation  (الأجور)
    row57: total intermediate inputs at basic prices
    row65: total output (used for verification only)

Column mapping: sector in IO row i maps to IO column (i - 6).
    e.g. ISIC13 is in data row 17 -> column 11

Two sector groupings (matching estimate_beta_s.py):

  GROUPING 1 — Baseline (2 sectors):
    T  : Textiles + Wearing apparel (ISIC 13, 14)
    O  : All other manufacturing (ISIC 10-12, 15-33)

  GROUPING 2 — Extended (5 sectors):
    T  : Textiles + Wearing apparel (ISIC 13, 14)
    S1 : Food products (ISIC 10)
    S2 : Chemicals (ISIC 20-21)
    S3 : Non-metallic minerals (ISIC 23)
    O  : All remaining manufacturing

Aggregation: wages and intermediates are summed across constituent rows before
taking the ratio, so alpha reflects the sector-aggregate cost share (output-
weighted across sub-sectors).

Units: millions of Egyptian pounds, 2008.
"""

import json
import os
import xlrd

IO_PATH = r"C:\Users\Admin\Desktop\Idea QIZs and Development\IO Tables\Egypt 2008 IO Tables.xls"
OUT_PATH = os.path.join(os.path.dirname(__file__), "params_estimated.json")

# ── IO row indices for each sector (0-based) ─────────────────────────────────
ROWS_T  = [17, 18]   # ISIC 13 textiles + 14 wearing apparel
ROWS_S1 = [14]       # ISIC 10 food products
ROWS_S2 = [23]       # ISIC 20-21 chemicals
ROWS_S3 = [25]       # ISIC 23 non-metallic minerals
ALL_MFG_ROWS = list(range(14, 33))
ROWS_O_BASE     = [r for r in ALL_MFG_ROWS if r not in ROWS_T]
ROWS_O_EXTENDED = [r for r in ALL_MFG_ROWS
                   if r not in ROWS_T + ROWS_S1 + ROWS_S2 + ROWS_S3]

# Column mapping: data row i -> data column (i - 6)
def row_to_col(row_idx: int) -> int:
    return row_idx - 6

# IO row indices for value-added block
ROW_WAGES  = 62   # labor compensation
ROW_INTERM = 57   # total intermediate inputs at basic prices
ROW_OUTPUT = 65   # total output (verification)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_io_sheet(path: str):
    wb = xlrd.open_workbook(path)
    return wb.sheet_by_name("IO _2")


def sector_labor_stats(sh, rows: list) -> dict:
    """Sum wages, intermediates, output across constituent IO rows."""
    wages  = sum(sh.cell_value(ROW_WAGES,  row_to_col(r)) for r in rows)
    interm = sum(sh.cell_value(ROW_INTERM, row_to_col(r)) for r in rows)
    output = sum(sh.cell_value(ROW_OUTPUT, row_to_col(r)) for r in rows)
    alpha  = wages / (wages + interm)
    return {
        "wages":        round(wages,  1),
        "intermediates": round(interm, 1),
        "output":       round(output, 1),
        "alpha":        round(alpha,  6),
    }


def compute_grouping(sh, sector_rows: dict) -> dict:
    """Given {sector_name: [row_indices]}, return stats and alpha for each."""
    return {s: sector_labor_stats(sh, rows) for s, rows in sector_rows.items()}


# ── main estimation ──────────────────────────────────────────────────────────

def estimate_alpha(path: str = IO_PATH) -> dict:
    sh = load_io_sheet(path)

    # Grouping 1: T + O
    g1 = compute_grouping(sh, {
        "T": ROWS_T,
        "O": ROWS_O_BASE,
    })

    # Grouping 2: T + S1 + S2 + S3 + S4 + O
    g2 = compute_grouping(sh, {
        "T":  ROWS_T,
        "S1": ROWS_S1,
        "S2": ROWS_S2,
        "S3": ROWS_S3,
        "O":  ROWS_O_EXTENDED,
    })

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
            "sectors": list(g1.keys()),
            "alpha":        {s: v["alpha"] for s, v in g1.items()},
            "diagnostics":  g1,
        },
        "grouping_2": {
            "description": "Extended: T + top-4 sectors by absorption + O (residual)",
            "sectors": list(g2.keys()),
            "alpha":        {s: v["alpha"] for s, v in g2.items()},
            "diagnostics":  g2,
        },
        "sector_labels": sector_labels,
        "source": "Egypt IO Table 2008, sheet 'IO _2'",
        "method": "alpha_s = Wages_s / (Wages_s + Intermediates_s); "
                  "wages = row62, intermediates = row57, col = row_index - 6",
        "units": "millions of Egyptian pounds, 2008",
    }


def update_params_file(result: dict, out_path: str = OUT_PATH):
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            params = json.load(f)
    else:
        params = {}
    params["alpha_s"] = result
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    return out_path


# ── reporting ────────────────────────────────────────────────────────────────

def print_grouping(name: str, g: dict, labels: dict):
    sep = "-" * 75
    print(f"\n  {sep}")
    print(f"  {name}: {g['description']}")
    print(f"  {sep}")
    print(f"  {'Sector':<6}  {'Label':<42}  {'Wages':>10}  {'Interm':>10}  {'alpha':>8}")
    print(f"  {sep}")
    for s in g["sectors"]:
        d = g["diagnostics"][s]
        lbl = labels.get(s, s)
        print(f"  {s:<6}  {lbl:<42}  {d['wages']:>10,.1f}  {d['intermediates']:>10,.1f}  {d['alpha']:>8.4f}")
    print(f"  {sep}")


if __name__ == "__main__":
    result = estimate_alpha()

    print("=" * 75)
    print("  alpha_s: labor share in variable production costs")
    print("=" * 75)

    for gname, gkey in [("Grouping 1 - Baseline", "grouping_1"),
                        ("Grouping 2 - Extended", "grouping_2")]:
        print_grouping(gname, result[gkey], result["sector_labels"])

    saved = update_params_file(result)
    print(f"\n  Saved to: {saved}")
