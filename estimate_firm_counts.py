#!/usr/bin/env python3
"""
estimate_firm_counts.py
-----------------------
Computes firm count ratios n_Q_s / n_N_s by sector from the Egypt
Industrial Statistics 2004 (pre-QIZ baseline). These ratios are the
target moments for calibrating relative entry costs f^E_rs.

QIZ region definition: 15 governorates from the official QIZ protocol
(IFTA Federal Register notices 69 FR 78094, 70 FR 69622, 74 FR 4482,
78 FR 15802). See Excel Sheet Data/QIZ Governorates by Treatment Year.xlsx.

Sector groupings mirror params_estimated.json:
  Grouping 1: T (Textiles + Wearing Apparel) vs O (other)
  Grouping 2: T / S1 (Food) / S2 (Chemicals) / S3 (Minerals) / O

Output: updates params_estimated.json with firm_count_ratio entry.

Source data: Data Egypt Industrial Stats 2004 (2).xlsx, Sheet2
"""

import json
import os
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

BASE = os.path.dirname(__file__)
DATA = r"C:\Users\Admin\Desktop\Data - Egypt\Data Egypt Industrial Stats 2004 (2).xlsx"
OUT  = os.path.join(BASE, "params_estimated.json")

# ── QIZ region definition ─────────────────────────────────────────────────────

QIZ_GOVS = [
    "Alexandria", "Cairo", "Port Said", "Suez",          # 2004
    "Giza", "Kalyoubia", "Dakhliaya", "Demiatte",        # 2005
    "Gharbiyyah", "Monofiyyah", "Ismailiyya", "Suez",    # 2005
    "Beni Suef", "Menya",                                 # 2009
    "Kafr Sheikh", "Beheira",                             # 2013
]
# Remove duplicates
QIZ_GOVS = list(dict.fromkeys(QIZ_GOVS))

# ── load data ─────────────────────────────────────────────────────────────────

df = pd.read_excel(DATA, sheet_name="Sheet2")
df["Gov_clean"] = df["Gov"].str.strip()
df["qiz"] = df["Gov_clean"].isin(QIZ_GOVS).astype(int)

# Check unmatched
unmatched = [g for g in QIZ_GOVS if g not in df["Gov_clean"].tolist()]
if unmatched:
    print(f"WARNING: unmatched QIZ govs: {unmatched}")

# ── sector groupings ──────────────────────────────────────────────────────────

# Grouping 1
df["sector_g1"] = "O"
df.loc[df["Industry"].isin(["Textile", "Wearing Apparel"]), "sector_g1"] = "T"

# Grouping 2
df["sector_g2"] = "O"
df.loc[df["Industry"].isin(["Textile", "Wearing Apparel"]),              "sector_g2"] = "T"
df.loc[df["Industry"].isin(["Food and Drinks"]),                          "sector_g2"] = "S1"
df.loc[df["Industry"].isin(["Chemicals", "Petrochemicals", "Rubber"]),    "sector_g2"] = "S2"
df.loc[df["Industry"].isin(["Other Minerals", "Main Minerals",
                             "Complex Minerals"]),                         "sector_g2"] = "S3"

# ── compute grouping 1 ────────────────────────────────────────────────────────

g1 = df.groupby(["qiz", "sector_g1"])["numestabl"].sum().reset_index()
g1w = g1.pivot(index="sector_g1", columns="qiz",
               values="numestabl").rename(columns={0: "N", 1: "Q"})
g1w["ratio_Q_to_N"] = g1w["Q"] / g1w["N"]

print("=" * 55)
print("  Grouping 1: T vs O")
print("=" * 55)
print(g1w.to_string())

# ── compute grouping 2 ────────────────────────────────────────────────────────

g2 = df.groupby(["qiz", "sector_g2"])["numestabl"].sum().reset_index()
g2w = g2.pivot(index="sector_g2", columns="qiz",
               values="numestabl").rename(columns={0: "N", 1: "Q"})
g2w["ratio_Q_to_N"] = g2w["Q"] / g2w["N"]

print()
print("=" * 55)
print("  Grouping 2: T / S1 / S2 / S3 / O")
print("=" * 55)
print(g2w.to_string())
print()
print(f"  Total non-QIZ firms: {int(df[df['qiz']==0]['numestabl'].sum())}")
print(f"  Total QIZ firms:     {int(df[df['qiz']==1]['numestabl'].sum())}")

# ── save ─────────────────────────────────────────────────────────────────────

if os.path.exists(OUT):
    with open(OUT, "r") as f:
        params = json.load(f)
else:
    params = {}

params["firm_count_ratio"] = {
    "description": "Ratio of firm counts QIZ/non-QIZ by sector (n_Q_s / n_N_s). Target moment for calibrating relative entry costs f^E_rs.",
    "qiz_definition": "15 governorates from official QIZ protocol",
    "qiz_govs": QIZ_GOVS,
    "non_qiz_govs": df[df["qiz"] == 0]["Gov_clean"].unique().tolist(),
    "grouping_1": {
        "n_N":          {s: int(g1w.loc[s, "N"]) for s in ["T", "O"]},
        "n_Q":          {s: int(g1w.loc[s, "Q"]) for s in ["T", "O"]},
        "ratio_Q_to_N": {s: round(g1w.loc[s, "ratio_Q_to_N"], 4) for s in ["T", "O"]},
    },
    "grouping_2": {
        "n_N":          {s: int(g2w.loc[s, "N"]) for s in ["T", "S1", "S2", "S3", "O"]},
        "n_Q":          {s: int(g2w.loc[s, "Q"]) for s in ["T", "S1", "S2", "S3", "O"]},
        "ratio_Q_to_N": {s: round(g2w.loc[s, "ratio_Q_to_N"], 4) for s in ["T", "S1", "S2", "S3", "O"]},
    },
    "sector_mapping": {
        "T":  "Textile + Wearing Apparel",
        "S1": "Food and Drinks",
        "S2": "Chemicals + Petrochemicals + Rubber",
        "S3": "Other Minerals + Main Minerals + Complex Minerals",
        "O":  "Remaining manufacturing"
    },
    "source": "Data Egypt Industrial Stats 2004 (2).xlsx, Sheet2. Establishment counts by 2-digit industry x governorate, 2004.",
    "note": "Kafr Sheikh and Beheira added to QIZ in 2013 (liberalized). For robustness, exclude these two and recompute."
}

with open(OUT, "w") as f:
    json.dump(params, f, indent=2)

print(f"\n  Saved to: {OUT}")
