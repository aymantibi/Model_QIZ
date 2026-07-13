#!/usr/bin/env python3
"""
estimate_price_wedge_reg.py
---------------------------
Estimates the Israeli vs. RoW intermediate input price wedge
p_IL,s / p_CN,s using a unit-value regression with fixed effects.

Regression specification:
    log(uv_iht) = alpha_i + alpha_h + alpha_t + beta_ISR * 1[ISR] + beta_IND * 1[IND] + eps

where i = firm (importer), h = HS6 product, t = year, origin in {ISR, CHN, IND}.
China (CHN) is the omitted reference category.
beta_ISR = log(p_ISR / p_CHN), exp(beta_ISR) = price ratio ISR/CHN.

Four specifications:
    Spec 1: product + year FE
    Spec 2: product x year FE
    Spec 3: firm + product + year FE
    Spec 4: firm + product x year FE

Sample: textile/apparel imports (HS 50-63), ISR + CHN + IND, pre-period 2005-2008.
Only firm-product-year cells where at least two origins are observed (overlap requirement).

Outputs:
    - Regression table printed to console
    - Results saved to params_estimated.json under "price_wedge_reg"
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

PRE_YEARS    = [2005, 2006, 2007, 2008]
ORIGINS      = {"ISR", "CHN", "IND"}
REFERENCE    = "CHN"
HS2_TEXTILE  = set(range(50, 64))   # HS chapters 50-63


# ── data loading ──────────────────────────────────────────────────────────────

def load_year(year: int) -> pd.DataFrame:
    folder = os.path.join(DATA_DIR, f"EID-Imports-{year} STATA")
    files  = glob.glob(os.path.join(folder, "*.dta"))
    if not files:
        raise FileNotFoundError(f"No .dta file for {year}")
    df, _ = pyreadstat.read_dta(files[0])
    return df


def load_textile_panel() -> pd.DataFrame:
    frames = []
    for yr in PRE_YEARS:
        df = load_year(yr)
        frames.append(df)
        print(f"  Loaded {yr}: {len(df):,} rows")
    raw = pd.concat(frames, ignore_index=True)

    # Keep only textile HS chapters
    raw["hs2"] = (raw["Product_HS6"].astype(str).str.zfill(6).str[:2]
                  .astype(int))
    raw = raw[raw["hs2"].isin(HS2_TEXTILE)].copy()

    # Keep only three origins
    raw = raw[raw["Cntry_Org_Code"].isin(ORIGINS)].copy()

    # Valid observations
    raw = raw[(raw["Quantity"] > 0) & (raw["ImpVal_USD"] > 0)].copy()
    raw["uv"]     = raw["ImpVal_USD"] / raw["Quantity"]
    raw["log_uv"] = np.log(raw["uv"])

    print(f"  Textile-apparel rows (ISR/CHN/IND): {len(raw):,}")
    for o in sorted(ORIGINS):
        n = (raw["Cntry_Org_Code"] == o).sum()
        print(f"    {o}: {n:,}")

    return raw


# ── sample construction ───────────────────────────────────────────────────────

def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to firm-HS6-year-origin medians, then keep only
    product-year cells where at least 2 origins are present
    (ensures within-product comparison).
    """
    agg = (df.groupby(["Trader_ID", "Product_HS6", "Year",
                        "Cntry_Org_Code"])["log_uv"]
             .median()
             .reset_index()
             .rename(columns={"Cntry_Org_Code": "origin"}))

    # Count distinct origins per (product, year) cell
    py_origins = (agg.groupby(["Product_HS6", "Year"])["origin"]
                     .nunique()
                     .rename("n_origins"))
    agg = agg.merge(py_origins, on=["Product_HS6", "Year"])
    agg = agg[agg["n_origins"] >= 2].drop(columns="n_origins")

    print(f"  After overlap filter: {len(agg):,} firm-product-year-origin obs")
    print(f"  Unique product-year cells: "
          f"{agg.groupby(['Product_HS6','Year']).ngroups:,}")
    return agg


# ── OLS with dummies (absorb FE) via pandas linalg ───────────────────────────

def ols_absorb(y: np.ndarray, X: np.ndarray, cols: list):
    """
    OLS via normal equations with heteroskedasticity-robust (HC1) SEs.
    Returns dict with coef, se, t, p for each column in cols.
    """
    n, k = X.shape
    XtX  = X.T @ X
    Xty  = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    resid = y - X @ beta
    # HC1 sandwich
    meat  = (X * resid[:, None]).T @ (X * resid[:, None])
    Vinv  = np.linalg.solve(XtX, np.eye(k))
    V     = (n / (n - k)) * Vinv @ meat @ Vinv
    se    = np.sqrt(np.diag(V))
    t     = beta / se

    from scipy import stats as sp_stats
    p = 2 * sp_stats.t.sf(np.abs(t), df=n - k)

    return {c: {"beta": beta[i], "se": se[i], "t": t[i], "p": p[i]}
            for i, c in enumerate(cols)}


# ── run one specification ──────────────────────────────────────────────────────

def run_spec(df: pd.DataFrame, spec_name: str, fe_cols: list):
    """
    fe_cols: list of column names in df to use as categorical FEs.
    Returns dict with ISR and IND coefficient info + N, R2.
    """
    sub = df.copy()

    # Create dummies for ISR and IND (CHN is reference)
    sub["d_ISR"] = (sub["origin"] == "ISR").astype(float)
    sub["d_IND"] = (sub["origin"] == "IND").astype(float)

    # Build FE dummy matrix using pandas get_dummies (sparse friendly)
    fe_parts = []
    for col in fe_cols:
        dums = pd.get_dummies(sub[col], prefix=col, drop_first=True)
        fe_parts.append(dums)

    fe_matrix = pd.concat(fe_parts, axis=1).astype(float)
    regressors = pd.concat([sub[["d_ISR", "d_IND"]], fe_matrix], axis=1)

    # Add constant? No — FEs absorb the intercept when using drop_first=True
    # but we need one group dropped per FE block; drop_first handles this.

    y = sub["log_uv"].values
    X = regressors.values
    cols = list(regressors.columns)

    result = ols_absorb(y, X, cols)

    # R-squared
    y_hat = X @ np.array([result[c]["beta"] for c in cols])
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "spec":    spec_name,
        "N":       len(sub),
        "R2":      round(r2, 4),
        "ISR":     {k: round(v, 5) for k, v in result["d_ISR"].items()},
        "IND":     {k: round(v, 5) for k, v in result["d_IND"].items()},
        "ratio_ISR_CHN": round(np.exp(result["d_ISR"]["beta"]), 4),
        "ratio_IND_CHN": round(np.exp(result["d_IND"]["beta"]), 4),
    }


# ── within-demean approach for high-dimensional FE ────────────────────────────

def demean_by(df: pd.DataFrame, cols: list, var: str) -> pd.Series:
    """Sequentially demean `var` within each group in cols (alternating projections)."""
    x = df[var].copy()
    for _ in range(50):    # iterate until convergence
        x_old = x.copy()
        for c in cols:
            x = x - df.groupby(c)[var].transform("mean") + x.mean()
        if (x - x_old).abs().max() < 1e-9:
            break
    return x


def run_spec_demean(df: pd.DataFrame, spec_name: str,
                    fe_groups: list, interact_cols: list = None):
    """
    Estimate using within-group demeaning for high-dim FE.
    fe_groups: list of column names to demean on.
    interact_cols: if not None, create a combined interaction key first.
    Returns same structure as run_spec.
    """
    sub = df.copy()
    sub["d_ISR"] = (sub["origin"] == "ISR").astype(float)
    sub["d_IND"] = (sub["origin"] == "IND").astype(float)

    groups = list(fe_groups)
    if interact_cols:
        sub["_interact"] = (sub[interact_cols[0]].astype(str) + "_"
                            + sub[interact_cols[1]].astype(str))
        groups = [g for g in groups if g not in interact_cols] + ["_interact"]

    # Demean all variables
    y_dm     = demean_by(sub, groups, "log_uv")
    d_isr_dm = demean_by(sub, groups, "d_ISR")
    d_ind_dm = demean_by(sub, groups, "d_IND")

    y  = y_dm.values
    X  = np.column_stack([d_isr_dm.values, d_ind_dm.values])

    n, k = X.shape
    XtX  = X.T @ X
    Xty  = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    resid = y - X @ beta

    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    Vinv = np.linalg.solve(XtX, np.eye(k))
    V    = (n / (n - k)) * Vinv @ meat @ Vinv
    se   = np.sqrt(np.diag(V))
    t    = beta / se

    from scipy import stats as sp_stats
    p_val = 2 * sp_stats.t.sf(np.abs(t), df=n - k)

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    def fmt(b, s, tv, pv):
        return {"beta": round(float(b), 5), "se": round(float(s), 5),
                "t": round(float(tv), 3), "p": round(float(pv), 4)}

    return {
        "spec":          spec_name,
        "N":             n,
        "R2":            round(r2, 4),
        "ISR":           fmt(beta[0], se[0], t[0], p_val[0]),
        "IND":           fmt(beta[1], se[1], t[1], p_val[1]),
        "ratio_ISR_CHN": round(float(np.exp(beta[0])), 4),
        "ratio_IND_CHN": round(float(np.exp(beta[1])), 4),
    }


# ── print results table ────────────────────────────────────────────────────────

def print_results(results: list):
    sep = "-" * 80
    print(f"\n{'='*80}")
    print("  Unit-value regression: log(uv_iht) = FE + beta_ISR*1[ISR] + beta_IND*1[IND]")
    print("  Reference origin: China (CHN).  Sample: HS 50-63, 2005-2008.")
    print(f"{'='*80}")
    print(f"  {'Spec':<30}  {'N':>7}  {'R2':>6}  "
          f"{'b_ISR':>8}  {'se_ISR':>7}  {'p_ISR':>6}  "
          f"{'b_IND':>8}  {'se_IND':>7}  {'p_IND':>6}  "
          f"{'p_ISR/p_CHN':>12}  {'p_IND/p_CHN':>12}")
    print(f"  {sep}")
    for r in results:
        print(f"  {r['spec']:<30}  {r['N']:>7,}  {r['R2']:>6.4f}  "
              f"{r['ISR']['beta']:>8.4f}  {r['ISR']['se']:>7.4f}  {r['ISR']['p']:>6.4f}  "
              f"{r['IND']['beta']:>8.4f}  {r['IND']['se']:>7.4f}  {r['IND']['p']:>6.4f}  "
              f"{r['ratio_ISR_CHN']:>12.4f}  {r['ratio_IND_CHN']:>12.4f}")
    print(f"  {sep}")
    print("  Interpretation: p_ISR/p_CHN < 1 => Israeli inputs cheaper than Chinese.")


def update_params_file(results: list, out_path: str = OUT_PATH):
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            params = json.load(f)
    else:
        params = {}
    params["price_wedge_reg"] = {
        "description": ("Unit-value regression log(uv) ~ FE + beta_ISR*1[ISR] + beta_IND*1[IND]. "
                        "Reference: China. Sample: HS 50-63, 2005-2008."),
        "specifications": results,
        "preferred":      results[-1],   # firm + product x year FE
    }
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading textile-apparel import panel (2005-2008)...")
    raw = load_textile_panel()

    print("Building regression sample...")
    df = build_sample(raw)

    results = []

    print("\nSpec 1: product + year FE...")
    r1 = run_spec_demean(df, "Spec1: product + year FE",
                         fe_groups=["Product_HS6", "Year"])
    results.append(r1)
    print(f"  beta_ISR={r1['ISR']['beta']:.4f} (se={r1['ISR']['se']:.4f}), "
          f"ratio={r1['ratio_ISR_CHN']:.4f}")

    print("Spec 2: product x year FE...")
    r2 = run_spec_demean(df, "Spec2: product x year FE",
                         fe_groups=["Product_HS6", "Year"],
                         interact_cols=["Product_HS6", "Year"])
    results.append(r2)
    print(f"  beta_ISR={r2['ISR']['beta']:.4f} (se={r2['ISR']['se']:.4f}), "
          f"ratio={r2['ratio_ISR_CHN']:.4f}")

    print("Spec 3: firm + product + year FE...")
    r3 = run_spec_demean(df, "Spec3: firm + product + year FE",
                         fe_groups=["Trader_ID", "Product_HS6", "Year"])
    results.append(r3)
    print(f"  beta_ISR={r3['ISR']['beta']:.4f} (se={r3['ISR']['se']:.4f}), "
          f"ratio={r3['ratio_ISR_CHN']:.4f}")

    print("Spec 4: firm + product x year FE...")
    r4 = run_spec_demean(df, "Spec4: firm + product x year FE",
                         fe_groups=["Trader_ID", "Product_HS6", "Year"],
                         interact_cols=["Product_HS6", "Year"])
    results.append(r4)
    print(f"  beta_ISR={r4['ISR']['beta']:.4f} (se={r4['ISR']['se']:.4f}), "
          f"ratio={r4['ratio_ISR_CHN']:.4f}")

    print_results(results)

    saved = update_params_file(results)
    print(f"\n  Saved to: {saved}")
