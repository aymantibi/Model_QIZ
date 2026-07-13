#!/usr/bin/env python3
"""
estimate_mfn_tariff.py
----------------------
Estimates value-weighted average US MFN tariff by sector, using:
  - Egypt's exports to the US by HS6, 2004 (UN Comtrade, US imports from Egypt)
  - US MFN applied tariff rates by HS6, 2004 (WTO IDB)

Formula:
    t_s = sum_{h in s} X_h * t_h / sum_{h in s} X_h

where X_h = Egypt's exports to US in product h, 2004 (USD)
      t_h = US MFN applied rate for product h, 2004 (percent)

Two sector groupings (same HS2 mapping as other estimation scripts):
    Grouping 1: T (HS 50-63), O (residual manufacturing)
    Grouping 2: T (HS 50-63), S1 (food HS 02,04,07-24),
                S2 (chemicals HS 28-38), S3 (non-metallic minerals HS 25,26,68-70),
                O (residual)
"""

import json
import os
import numpy as np
import pandas as pd

TRADE_PATH  = (r"C:\Users\Admin\Desktop\Idea QIZs and Development"
               r"\Trade Data Un\DataJobID-3063992_3063992_EgyptianExports2004.csv")
TARIFF_PATH = (r"C:\Users\Admin\Desktop\Idea QIZs and Development"
               r"\Export and Import Data Egypt\us_MFN_6products.csv")
OUT_PATH    = os.path.join(os.path.dirname(__file__), "params_estimated.json")

# ── sector HS2 mappings ───────────────────────────────────────────────────────
HS2_T  = set(range(50, 64))
HS2_S1 = {2, 4} | set(range(7, 25))
HS2_S2 = set(range(28, 39))
HS2_S3 = {25, 26, 68, 69, 70}
HS2_MFG = set(range(2, 98)) - {27}   # manufacturing, drop petroleum

SECTORS_G1 = {
    "T": HS2_T,
    "O": HS2_MFG - HS2_T,
}
SECTORS_G2 = {
    "T":  HS2_T,
    "S1": HS2_S1,
    "S2": HS2_S2,
    "S3": HS2_S3,
    "O":  HS2_MFG - HS2_T - HS2_S1 - HS2_S2 - HS2_S3,
}
SECTOR_LABELS = {
    "T":  "Textiles + Wearing apparel (HS 50-63)",
    "S1": "Food products (HS 02,04,07-24)",
    "S2": "Chemicals (HS 28-38)",
    "S3": "Non-metallic minerals (HS 25,26,68-70)",
    "O":  "Other manufacturing (residual)",
}


# ── load data ─────────────────────────────────────────────────────────────────

def load_trade(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # keep 2004, rename columns
    df = df[df["Year"] == 2004].copy()
    df["hs6"] = df["ProductCode"].astype(str).str.zfill(6)
    df["hs2"] = df["hs6"].str[:2].astype(int)
    df["trade_usd"] = df["TradeValue in 1000 USD"] * 1000
    return df[["hs6", "hs2", "trade_usd"]].copy()


def load_tariff(path: str, year: int = 2005) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["year"] == year].copy()
    df["hs6"] = df["product_code"].astype(str).str.zfill(6)
    df["tariff"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["hs6", "tariff"]].dropna().copy()


# ── weighted average tariff by sector ────────────────────────────────────────

def sector_tariff(merged: pd.DataFrame, hs2_codes: set) -> dict:
    sub = merged[merged["hs2"].isin(hs2_codes)].copy()
    if len(sub) == 0 or sub["trade_usd"].sum() == 0:
        return {"n_products": 0, "tariff_wavg": np.nan,
                "tariff_simple": np.nan, "trade_usd": 0}
    wavg   = (sub["tariff"] * sub["trade_usd"]).sum() / sub["trade_usd"].sum()
    simple = sub["tariff"].mean()
    return {
        "n_products":    len(sub),
        "trade_usd":     round(sub["trade_usd"].sum(), 0),
        "tariff_wavg":   round(wavg,   4),
        "tariff_simple": round(simple, 4),
        # ratio form for model: tau_s = 1 + t_s/100
        "tau":           round(1 + wavg / 100, 6),
    }


def estimate_grouping(merged: pd.DataFrame, sector_map: dict) -> dict:
    return {s: sector_tariff(merged, codes)
            for s, codes in sector_map.items()}


# ── output ────────────────────────────────────────────────────────────────────

def print_grouping(name: str, g: dict):
    sep = "-" * 75
    print(f"\n  {sep}")
    print(f"  {name}")
    print(f"  {sep}")
    print(f"  {'Sector':<6}  {'Label':<38}  {'N prods':>7}  "
          f"{'Trade USD':>14}  {'Wavg t%':>8}  {'Simple t%':>10}  {'tau':>7}")
    print(f"  {sep}")
    for s, d in g.items():
        lbl = SECTOR_LABELS.get(s, s)
        if d["n_products"] == 0:
            print(f"  {s:<6}  {lbl:<38}  {'N/A':>7}")
            continue
        print(f"  {s:<6}  {lbl:<38}  {d['n_products']:>7,}  "
              f"{d['trade_usd']:>14,.0f}  {d['tariff_wavg']:>7.2f}%  "
              f"{d['tariff_simple']:>9.2f}%  {d['tau']:>7.4f}")
    print(f"  {sep}")
    total_trade = sum(d["trade_usd"] for d in g.values() if d["n_products"] > 0)
    print(f"  Total Egypt-US trade in sample: ${total_trade:,.0f}")


def update_params_file(result: dict, out_path: str = OUT_PATH):
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            params = json.load(f)
    else:
        params = {}
    params["mfn_tariff"] = result
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    return out_path


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading trade data (US imports from Egypt, 2004)...")
    trade = load_trade(TRADE_PATH)
    print(f"  {len(trade):,} HS6 products, total trade: ${trade['trade_usd'].sum():,.0f}")

    print("Loading US MFN tariff rates (2004)...")
    tariff = load_tariff(TARIFF_PATH, year=2004)
    print(f"  {len(tariff):,} HS6 products with tariff rates")

    print("Merging...")
    merged = trade.merge(tariff, on="hs6", how="inner")
    print(f"  {len(merged):,} products matched (have both trade value and tariff)")
    print(f"  Unmatched trade value: "
          f"${trade[~trade['hs6'].isin(merged['hs6'])]['trade_usd'].sum():,.0f} "
          f"({100*(1 - merged['trade_usd'].sum()/trade['trade_usd'].sum()):.1f}% of total)")

    print("Estimating value-weighted MFN tariffs by sector...")
    g1 = estimate_grouping(merged, SECTORS_G1)
    g2 = estimate_grouping(merged, SECTORS_G2)

    print("\n" + "=" * 75)
    print("  t_s^MFN: value-weighted US MFN tariff by sector")
    print("  Weights: Egypt exports to US, 2004 (UN Comtrade)")
    print("  Rates:   US MFN applied rates, 2004 (WTO IDB)")
    print("=" * 75)
    print_grouping("Grouping 1 - Baseline", g1)
    print_grouping("Grouping 2 - Extended", g2)

    result = {
        "description": ("Value-weighted US MFN applied tariff by sector. "
                         "Weights = Egypt exports to US in 2004 (pre-QIZ). "
                         "Rates = WTO IDB MFN applied rates 2004."),
        "grouping_1": {
            "tau":           {s: v["tau"]         for s, v in g1.items()},
            "tariff_wavg":   {s: v["tariff_wavg"] for s, v in g1.items()},
            "details":       g1,
        },
        "grouping_2": {
            "tau":           {s: v["tau"]         for s, v in g2.items()},
            "tariff_wavg":   {s: v["tariff_wavg"] for s, v in g2.items()},
            "details":       g2,
        },
        "sector_labels": SECTOR_LABELS,
        "source": "UN Comtrade (US imports from Egypt 2004) + WTO IDB MFN rates 2004",
        "note": "tau_s = 1 + t_s/100 is the iceberg tariff factor used in the model",
    }

    saved = update_params_file(result)
    print(f"\n  Saved to: {saved}")
