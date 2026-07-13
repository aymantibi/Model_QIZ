#!/usr/bin/env python3
"""
estimate_price_wedge.py
-----------------------
Estimates the Israeli vs. rest-of-world intermediate input price wedge
p_IL,s / p_RW,s for each sector, from Egyptian customs import data.

Method (from appendix):
    uv_{o,h,t} = ImpVal_USD_{o,h,t} / Quantity_{o,h,t}

    log(p_IL,s / p_RW,s) = median_{h in s, t} [log(uv_IL,h,t) - log(uv_RW,h,t)]

where h indexes HS6 products, t indexes years, o indexes origin country.

The comparison is made within narrow product-year cells (same HS6, same year)
to eliminate composition bias. Only products where both Israeli and RoW
observations exist and use the same quantity unit are included.

Steps:
    1. Load raw annual import files (EID-Imports-YYYY) for pre-period 2005-2008
    2. Identify Israeli imports (Cntry_Org_Code == 'ISR')
    3. Keep only products where Israel and RoW use the same quantity unit
    4. Compute unit values: uv = ImpVal_USD / Quantity
    5. Winsorize unit values at 1st-99th percentile within each product-year
    6. Compute median unit value by product-year-origin (IL vs RoW)
    7. Take log ratio within each product-year cell
    8. Aggregate to sector level using median across product-year cells

Sector mapping (HS2 chapters -> model sectors, ISIC Rev3):
    T  : HS 50-63  (textiles and apparel)
    S1 : HS 02,04,07-24  (food products, approx ISIC 15)
    S2 : HS 28-38  (chemicals, approx ISIC 24)
    S3 : HS 25,26,68-70  (non-metallic minerals, approx ISIC 26)
    O  : all remaining manufacturing HS chapters

Two sector groupings:
    Grouping 1: T, O
    Grouping 2: T, S1, S2, S3, O
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import pyreadstat

DATA_DIR = (r"C:\Users\Admin\Desktop\Idea QIZs and Development"
            r"\Export and Import Data Egypt")
OUT_PATH = os.path.join(os.path.dirname(__file__), "params_estimated.json")

# Pre-period years
PRE_YEARS = [2005, 2006, 2007, 2008]

# Israel country code
ISRAEL_CODE = "ISR"

# HS2 chapter -> sector mapping
# T: textiles and apparel (HS 50-63)
HS2_T  = set(range(50, 64))
# S1: food (HS 02,04,07-24)
HS2_S1 = {2, 4} | set(range(7, 25))
# S2: chemicals (HS 28-38)
HS2_S2 = set(range(28, 39))
# S3: non-metallic minerals (HS 25,26,68-70)
HS2_S3 = {25, 26, 68, 69, 70}
# All manufacturing HS chapters (approx)
HS2_MFG = set(range(2, 98)) - {3, 5, 14, 24, 27}  # exclude agri raw, fish, etc.

SECTOR_LABELS = {
    "T":  "Textiles + Wearing apparel (HS 50-63)",
    "S1": "Food products (HS 02,04,07-24)",
    "S2": "Chemicals (HS 28-38)",
    "S3": "Non-metallic minerals (HS 25,26,68-70)",
    "O":  "Other manufacturing (residual)",
}

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


# ── data loading ─────────────────────────────────────────────────────────────

def load_year(year: int) -> pd.DataFrame:
    folder = os.path.join(DATA_DIR, f"EID-Imports-{year} STATA")
    files  = glob.glob(os.path.join(folder, "*.dta"))
    if not files:
        raise FileNotFoundError(f"No .dta file found for year {year}")
    df, _ = pyreadstat.read_dta(files[0])
    return df


def load_preperiod(years: list) -> pd.DataFrame:
    frames = []
    for yr in years:
        df = load_year(yr)
        frames.append(df)
        print(f"  Loaded {yr}: {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


# ── unit value computation ────────────────────────────────────────────────────

def compute_uv(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each observation compute unit value = ImpVal_USD / Quantity.
    Flag Israeli observations.
    Add HS2 chapter.
    """
    df = df[df["Quantity"] > 0].copy()
    df["is_israel"] = df["Cntry_Org_Code"] == ISRAEL_CODE
    df["uv"]  = df["ImpVal_USD"] / df["Quantity"]
    df["hs2"] = (df["Product_HS6"].astype(str).str.zfill(6).str[:2]
                 .astype(int))
    return df[df["uv"] > 0]


def filter_matching_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only product-year cells where Israeli and RoW observations
    use the same (modal) quantity unit.
    """
    # Modal unit by product-year-origin
    isr = (df[df["is_israel"]]
           .groupby(["Product_HS6", "Year"])["Qunt_Unit"]
           .agg(lambda x: x.mode()[0])
           .rename("unit_isr"))
    row = (df[~df["is_israel"]]
           .groupby(["Product_HS6", "Year"])["Qunt_Unit"]
           .agg(lambda x: x.mode()[0])
           .rename("unit_row"))
    units = pd.concat([isr, row], axis=1).dropna()
    same  = units[units["unit_isr"] == units["unit_row"]].index
    return df.set_index(["Product_HS6", "Year"]).loc[
        df.set_index(["Product_HS6", "Year"]).index.isin(same)
    ].reset_index()


def winsorize_uv(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize unit values at 1st-99th percentile within product-year."""
    lo = df.groupby(["Product_HS6", "Year"])["uv"].transform(
         lambda x: x.quantile(0.01))
    hi = df.groupby(["Product_HS6", "Year"])["uv"].transform(
         lambda x: x.quantile(0.99))
    return df[(df["uv"] >= lo) & (df["uv"] <= hi)].copy()


# ── opportunity cost estimator ────────────────────────────────────────────────

def compute_opportunity_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each product-year cell where Israel is observed, compute:
        premium_ht = (uv_ISR,ht - min_uv_nonISR,ht) / min_uv_nonISR,ht * 100

    This is the percentage by which the Israeli unit value exceeds the
    cheapest available global alternative in the same product-year cell.

    Returns DataFrame with columns: Product_HS6, Year, hs2,
        uv_isr, uv_cheapest, premium_pct.
    """
    isr_uv = (df[df["is_israel"]]
              .groupby(["Product_HS6", "Year", "hs2"])["uv"]
              .median()
              .rename("uv_isr"))
    # cheapest credible non-Israeli supplier: 5th percentile (avoids single
    # misdeclared or near-zero shipments that would inflate the premium)
    cheapest = (df[~df["is_israel"]]
                .groupby(["Product_HS6", "Year", "hs2"])["uv"]
                .quantile(0.05)
                .rename("uv_cheapest"))
    oc = pd.concat([isr_uv, cheapest], axis=1).dropna().reset_index()
    oc["premium_pct"] = (oc["uv_isr"] - oc["uv_cheapest"]) / oc["uv_cheapest"] * 100
    return oc


def sector_opportunity_cost(oc: pd.DataFrame, hs2_codes: set) -> dict:
    """Aggregate opportunity cost premium to sector level using median."""
    sub = oc[oc["hs2"].isin(hs2_codes)]
    if len(sub) == 0:
        return {"n_products": 0, "median_premium_pct": np.nan,
                "mean_premium_pct": np.nan}
    return {
        "n_obs":             len(sub),
        "n_products":        sub["Product_HS6"].nunique(),
        "median_premium_pct": round(sub["premium_pct"].median(), 1),
        "mean_premium_pct":   round(sub["premium_pct"].mean(),   1),
        "pct_IL_higher":      round((sub["premium_pct"] > 0).mean() * 100, 1),
        # ratio form for model: p_IL / p_cheapest
        "ratio":              round(1 + sub["premium_pct"].median() / 100, 4),
    }


def estimate_oc_grouping(oc: pd.DataFrame, sector_map: dict) -> dict:
    return {s: sector_opportunity_cost(oc, codes)
            for s, codes in sector_map.items()}


# ── log ratio estimation (legacy) ────────────────────────────────────────────

def compute_log_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each product-year cell, compute:
        log_ratio = log(median_uv_IL) - log(median_uv_RW)
    Returns DataFrame with columns: Product_HS6, Year, hs2, log_ratio.
    """
    isr_uv = (df[df["is_israel"]]
              .groupby(["Product_HS6", "Year", "hs2"])["uv"]
              .median()
              .rename("uv_isr"))
    row_uv = (df[~df["is_israel"]]
              .groupby(["Product_HS6", "Year", "hs2"])["uv"]
              .median()
              .rename("uv_row"))
    uv = pd.concat([isr_uv, row_uv], axis=1).dropna().reset_index()
    uv["log_ratio"] = np.log(uv["uv_isr"] / uv["uv_row"])
    return uv


def sector_wedge(log_ratios: pd.DataFrame, hs2_codes: set) -> dict:
    """Aggregate log ratios to sector level."""
    sub = log_ratios[log_ratios["hs2"].isin(hs2_codes)]
    if len(sub) == 0:
        return {"n_products": 0, "log_ratio": np.nan, "ratio": np.nan}
    med = sub["log_ratio"].median()
    return {
        "n_obs":      len(sub),
        "n_products": sub["Product_HS6"].nunique(),
        "log_ratio":  round(med, 4),
        "ratio":      round(np.exp(med), 4),
        "pct_IL_higher": round((sub["log_ratio"] > 0).mean() * 100, 1),
    }


def estimate_grouping(log_ratios: pd.DataFrame, sector_map: dict) -> dict:
    return {s: sector_wedge(log_ratios, codes)
            for s, codes in sector_map.items()}


# ── output and saving ─────────────────────────────────────────────────────────

def update_params_file(result: dict, out_path: str = OUT_PATH):
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            params = json.load(f)
    else:
        params = {}
    params["price_wedge"] = result
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    return out_path


def print_grouping(name: str, g: dict):
    sep = "-" * 75
    print(f"\n  {sep}")
    print(f"  {name}")
    print(f"  {sep}")
    print(f"  {'Sector':<6}  {'Label':<38}  {'N obs':>6}  "
          f"{'log ratio':>10}  {'ratio':>8}  {'% IL>RW':>8}")
    print(f"  {sep}")
    for s, d in g.items():
        lbl = SECTOR_LABELS.get(s, s)
        if d["n_obs"] == 0:
            print(f"  {s:<6}  {lbl:<38}  {'N/A':>6}")
            continue
        print(f"  {s:<6}  {lbl:<38}  {d['n_obs']:>6,}  "
              f"{d['log_ratio']:>10.4f}  {d['ratio']:>8.4f}  "
              f"{d['pct_IL_higher']:>7.1f}%")
    print(f"  {sep}")
    print(f"\n  Interpretation: ratio < 1 means Israeli inputs are cheaper than RoW.")
    print(f"  ratio > 1 means Israeli inputs are more expensive than RoW.")


if __name__ == "__main__":
    print("Loading pre-period import data (2005-2008)...")
    raw = load_preperiod(PRE_YEARS)
    print(f"  Total rows loaded: {len(raw):,}")

    print("Computing unit values...")
    df = compute_uv(raw)
    print(f"  Rows with valid UV: {len(df):,}")
    print(f"  Israeli obs: {df['is_israel'].sum():,}")

    print("Filtering to matching quantity units...")
    df = filter_matching_units(df)
    print(f"  Rows after unit filter: {len(df):,}")

    print("Winsorizing unit values...")
    df = winsorize_uv(df)
    print(f"  Rows after winsorizing: {len(df):,}")

    print("Computing opportunity cost premium (Israeli vs. cheapest global)...")
    oc = compute_opportunity_cost(df)
    print(f"  Product-year cells with Israeli obs: {len(oc):,}")

    print("Computing log ratios (legacy: Israeli vs. median RoW)...")
    log_ratios = compute_log_ratios(df)
    print(f"  Product-year cells with both IL and RoW: {len(log_ratios):,}")

    print("Aggregating to sectors...")
    oc_g1 = estimate_oc_grouping(oc, SECTORS_G1)
    oc_g2 = estimate_oc_grouping(oc, SECTORS_G2)
    g1    = estimate_grouping(log_ratios, SECTORS_G1)
    g2    = estimate_grouping(log_ratios, SECTORS_G2)

    # ── print opportunity cost results ────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  OPPORTUNITY COST: Israeli vs. cheapest global alternative")
    print("  premium_pct = (uv_ISR - min_uv_nonISR) / min_uv_nonISR * 100")
    print("=" * 75)

    for name, oc_g in [("Grouping 1 - Baseline", oc_g1),
                       ("Grouping 2 - Extended", oc_g2)]:
        sep = "-" * 75
        print(f"\n  {sep}")
        print(f"  {name}")
        print(f"  {sep}")
        print(f"  {'Sector':<6}  {'Label':<38}  {'N obs':>6}  "
              f"{'Median %':>9}  {'Mean %':>8}  {'% IL>cheapest':>14}  {'ratio':>7}")
        print(f"  {sep}")
        for s, d in oc_g.items():
            lbl = SECTOR_LABELS.get(s, s)
            if d["n_obs"] == 0:
                print(f"  {s:<6}  {lbl:<38}  {'N/A':>6}")
                continue
            print(f"  {s:<6}  {lbl:<38}  {d['n_obs']:>6,}  "
                  f"{d['median_premium_pct']:>8.1f}%  "
                  f"{d['mean_premium_pct']:>7.1f}%  "
                  f"{d['pct_IL_higher']:>13.1f}%  "
                  f"{d['ratio']:>7.4f}")
        print(f"  {sep}")
    print("\n  Interpretation: median_premium_pct > 0 => Israeli inputs more expensive.")
    print("  Survey benchmark: 28 firms report >20% premium, 11 firms 10-20%, 7 firms <10%.")
    print("  Survey-implied median premium: ~20-25%.")

    # ── print legacy log-ratio results ────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  LEGACY: Israeli vs. median RoW (log ratio method)")
    print("=" * 75)
    print_grouping("Grouping 1 - Baseline", g1)
    print_grouping("Grouping 2 - Extended", g2)

    # ── save ──────────────────────────────────────────────────────────────────
    result = {
        "primary_source": "Survey (ECES 2006): median firm reports >20% Israeli premium",
        "primary_estimate": {
            "T":  {"ratio": 1.20, "premium_pct": 20.0,
                   "note": "Survey midpoint; 28/49 firms report >20%, 11 report 10-20%"},
            "S1": {"ratio": 1.20, "premium_pct": 20.0, "note": "Survey, no sector breakdown"},
            "S2": {"ratio": 1.20, "premium_pct": 20.0, "note": "Survey, no sector breakdown"},
            "S3": {"ratio": 1.20, "premium_pct": 20.0, "note": "Survey, no sector breakdown"},
            "O":  {"ratio": 1.20, "premium_pct": 20.0, "note": "Survey, no sector breakdown"},
        },
        "opportunity_cost_estimator": {
            "description": ("(uv_ISR,ht - min_uv_nonISR,ht) / min_uv_nonISR,ht * 100. "
                            "Median across product-year cells within sector. "
                            "Uses same cleaned sample as legacy estimator."),
            "grouping_1": oc_g1,
            "grouping_2": oc_g2,
        },
        "legacy_estimator": {
            "description": ("Median log(uv_ISR/uv_RoW) within product-year cells. "
                            "Note: unit value noise likely biases this downward."),
            "grouping_1": g1,
            "grouping_2": g2,
        },
        "sector_labels": SECTOR_LABELS,
        "source": "Egyptian customs import data EID-Imports 2005-2008 + ECES firm survey 2006",
    }

    saved = update_params_file(result)
    print(f"\n  Saved to: {saved}")
