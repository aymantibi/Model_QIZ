#!/usr/bin/env python3
"""
estimate_labor_endowments.py
----------------------------
Computes regional labor endowments L, L_Q, L_N from the Egypt Annual
Labor Force Survey 2004 (CAPMAS), aggregated by QIZ vs non-QIZ governorate.

QIZ region definition: 15 governorates from the official QIZ protocol
(IFTA Federal Register notices 69 FR 78094, 70 FR 69622, 74 FR 4482,
78 FR 15802). See Excel Sheet Data/QIZ Governorates by Treatment Year.xlsx.

Output: updates params_estimated.json with labor_endowments entry.
"""

import json
import os
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "Labor Force by Governorate 2004.xlsx")
OUT     = os.path.join(BASE, "params_estimated.json")

# ── QIZ region definition ─────────────────────────────────────────────────────

QIZ_GOVS = [
    "Cairo", "Alexandria", "Port Said", "Suez",
    "Demiatte", "Dahakliyya", "Kalioubia",
    "Kafr Sheikh", "Gharbiyya", "Monoufiyya", "Beheira",
    "Ismailiyya", "Giza", "Beni Soueif", "Menya"
]

# ── load data ─────────────────────────────────────────────────────────────────

df = pd.read_excel(DATA, sheet_name="Sheet1")[["Gov", "Labor Force"]].dropna()
df["Gov"] = df["Gov"].str.strip()

df["qiz"] = df["Gov"].isin(QIZ_GOVS).astype(int)

# Check for unmatched
unmatched = [g for g in QIZ_GOVS if g not in df["Gov"].tolist()]
if unmatched:
    print(f"WARNING: unmatched QIZ govs: {unmatched}")

# ── compute ───────────────────────────────────────────────────────────────────

L_Q = int(df[df["qiz"] == 1]["Labor Force"].sum())
L_N = int(df[df["qiz"] == 0]["Labor Force"].sum())
L   = int(df["Labor Force"].sum())

print("=" * 55)
print("  Labor Endowments — Egypt LFS 2004")
print("=" * 55)
print(f"  QIZ govs:     {df[df['qiz']==1]['Gov'].tolist()}")
print(f"  Non-QIZ govs: {df[df['qiz']==0]['Gov'].tolist()}")
print()
print(f"  L_Q = {L_Q:>12,}  ({L_Q/L:.4f})")
print(f"  L_N = {L_N:>12,}  ({L_N/L:.4f})")
print(f"  L   = {L:>12,}")
print(f"  L_Q / L_N = {L_Q/L_N:.4f}")
print("=" * 55)

# ── save ─────────────────────────────────────────────────────────────────────

if os.path.exists(OUT):
    with open(OUT, "r") as f:
        params = json.load(f)
else:
    params = {}

params["labor_endowments"] = {
    "description": "Total labor force by region (QIZ vs non-QIZ), 2004 pre-QIZ baseline.",
    "year": 2004,
    "L":   L,
    "L_Q": L_Q,
    "L_N": L_N,
    "L_Q_share": round(L_Q / L, 4),
    "L_N_share": round(L_N / L, 4),
    "L_Q_over_L_N": round(L_Q / L_N, 4),
    "qiz_govs": QIZ_GOVS,
    "non_qiz_govs": df[df["qiz"] == 0]["Gov"].tolist(),
    "source": "Egypt Annual Labor Force Survey 2004, CAPMAS. Pages 50-58.",
    "file": "Labor Force by Governorate 2004.xlsx",
    "note": "Units: persons. L_Q/L_N = 2.827 reflects that QIZ region covers major urban and Delta governorates."
}

with open(OUT, "w") as f:
    json.dump(params, f, indent=2)

print(f"\n  Saved to: {OUT}")
