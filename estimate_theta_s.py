#!/usr/bin/env python3
"""
estimate_theta_s.py
-------------------
Estimates the Pareto shape parameter theta_s for each sector.

Theory:
    Under Pareto productivity (shape theta) and CES demand (elasticity sigma),
    firm export sales follow a Pareto distribution in the upper tail with shape:

        xi = theta / (sigma - 1)

    So:  theta = xi * (sigma - 1)

    We estimate xi from the upper tail of the firm export size distribution,
    then multiply by (sigma - 1) to recover theta.

Sample construction:
    - T (textiles+apparel): 2005 only — cleanest pre-treatment snapshot,
      avoids anticipation effects in the treated sector
    - All other sectors: estimate xi separately for each year 2005-2008,
      then average xi across years for stability

    For each year x sector cell: keep only firms with exp_world > 0.

Estimators:
    1. Hill (1975) MLE — corrected formula using x_(k+1) as threshold:
           xi_Hill = 1 / mean(log(x_(i) / x_(k+1)))  for i = 1..k
    2. OLS log-rank / log-size with Blom (rank - 0.5) correction:
           log(rank_i - 0.5) = a - xi * log(x_i)

    Primary xi: average of Hill and OLS at the 5% tail cutoff.
    Rationale: the Hill plot shows the Pareto slope stabilizes among the top
    5% of exporters; including the 5-10% range pulls in non-Pareto firms and
    biases xi downward. Robustness reported at 10% and 20% as well.
    Hill plot: xi estimated across k = 10..N/2 to identify stable region.

Conversion to theta:
    theta_s = xi_s * (sigma_s - 1)
    using literature sigma values supplied below.

Sector groupings (ISIC Rev 3 2-digit, sector2d variable):
    GROUPING 1 — Baseline (2 sectors):
        T : {17, 18}   textiles + wearing apparel
        O : all other manufacturing {15,19-36}

    GROUPING 2 — Extended (5 sectors):
        T  : {17, 18}  textiles + wearing apparel
        S1 : {15}      food products
        S2 : {24}      chemicals
        S3 : {26}      non-metallic minerals
        O  : residual manufacturing

Requirement: theta_s > sigma_s - 1 for finite mean sales.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = (r"C:\Users\Admin\Desktop\Idea QIZs and Development"
             r"\Export and Import Data Egypt\final_data_matched.dta")
OUT_PATH   = os.path.join(os.path.dirname(__file__), "params_estimated.json")
PLOT_DIR   = os.path.dirname(__file__)

# Sample years by sector type
# T (treated sector): 2005 only to avoid anticipation effects
YEARS_T   = [2005]
# All other sectors: year-by-year 2005-2008, then average xi across years
YEARS_O   = list(range(2005, 2009))

# Sigma values from Imbs and Mejean (2015) / literature
# T: textiles+apparel, S1: food, S2: chemicals, S3: non-metallic minerals
SIGMA = {
    "T":  6.7,
    "S1": 5.0,
    "S2": 6.2,
    "S3": 6.0,
    "O":  6.0,
}

# Tail cutoffs for robustness
TAIL_CUTOFFS   = [0.05, 0.10, 0.20]
PRIMARY_CUTOFF = 0.05   # 5% tail: uses only the top 5% of exporters by size,
                        # capturing the true Pareto region more cleanly

# Floor: if xi < 1 (theta would violate theta > sigma-1), set theta = sigma.
# Applied to residual O sectors where the Pareto assumption is weakest.
APPLY_FLOOR = True

# Sector groupings
MFG_2D = set(range(15, 37))
SECTORS_G1 = {
    "T": {17, 18},
    "O": MFG_2D - {17, 18},
}
SECTORS_G2 = {
    "T":  {17, 18},
    "S1": {15},
    "S2": {24},
    "S3": {26},
    "O":  MFG_2D - {17, 18, 15, 24, 26},
}
SECTOR_LABELS = {
    "T":  "Textiles + Wearing apparel (ISIC 17-18)",
    "S1": "Food products (ISIC 15)",
    "S2": "Chemicals (ISIC 24)",
    "S3": "Non-metallic minerals (ISIC 26)",
    "O":  "Other manufacturing (residual)",
}


# ── data preparation ─────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df, _ = pyreadstat.read_dta(path)
    df["sector2d"] = df["sector2d"].astype("Int64")
    return df


def get_active_exports(df: pd.DataFrame, codes: set, year: int) -> np.ndarray:
    """Active exporters in a single year x sector cell."""
    mask = (df["Year"] == year) & df["sector2d"].isin(codes) & (df["exp_world"] > 0)
    return df.loc[mask, "exp_world"].values.astype(float)


# ── estimators ───────────────────────────────────────────────────────────────

def hill_estimator(x: np.ndarray, k: int) -> float:
    """
    Corrected Hill (1975) MLE estimator.
    Uses x_(k+1) as the threshold (the value just below the tail).
    x must be sorted descending.
    """
    if k + 1 > len(x):
        return np.nan
    x_tail      = x[:k]
    x_threshold = x[k]          # x_(k+1), the first excluded value
    if x_threshold <= 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi = 1.0 / np.mean(np.log(x_tail / x_threshold))
    return float(xi)


def ols_logrank(x: np.ndarray, k: int) -> float:
    """
    OLS log-rank / log-size with Blom (rank - 0.5) correction.
    log(rank_i - 0.5) = a - xi * log(x_i)
    Returns xi = -slope.
    x must be sorted descending.
    """
    x_tail = x[:k]
    ranks  = np.arange(1, k + 1) - 0.5      # Blom correction
    log_x  = np.log(x_tail)
    log_r  = np.log(ranks)
    slope  = np.polyfit(log_x, log_r, 1)[0]
    return float(-slope)


def hill_plot_data(x: np.ndarray) -> tuple:
    """
    Compute Hill estimate for k = 10 to N//2.
    Returns (k_values, xi_values) for plotting.
    """
    x     = np.sort(x)[::-1]
    n     = len(x)
    k_min = 10
    k_max = n // 2
    ks    = np.arange(k_min, k_max + 1)
    xis   = np.array([hill_estimator(x, k) for k in ks])
    return ks, xis


def estimate_single_cross_section(x: np.ndarray, sigma: float) -> dict:
    """
    Estimate xi and theta from one cross-section (one year x sector).
    Returns dict with xi and robustness breakdown.
    """
    x = np.sort(x)[::-1]
    n = len(x)
    robust = {}
    for frac in TAIL_CUTOFFS:
        k   = max(10, int(np.floor(frac * n)))
        key = f"{int(frac*100)}pct"
        h   = hill_estimator(x, k)
        o   = ols_logrank(x, k)
        avg = 0.5 * (h + o) if not (np.isnan(h) or np.isnan(o)) else np.nan
        robust[key] = {
            "k":    k,
            "hill": round(h,   4) if not np.isnan(h) else None,
            "ols":  round(o,   4) if not np.isnan(o) else None,
            "xi":   round(avg, 4) if not np.isnan(avg) else None,
        }
    xi_primary = robust[f"{int(PRIMARY_CUTOFF*100)}pct"]["xi"]
    return {"n_firms": n, "xi": xi_primary, "robustness": robust}


def estimate_sector(df: pd.DataFrame, codes: set,
                    years: list, sigma: float) -> dict:
    """
    Estimate xi year-by-year then average across years.
    For single-year sectors (e.g. T with years=[2005]),
    this just returns the single-year estimate.
    """
    yearly = {}
    for yr in years:
        x = get_active_exports(df, codes, yr)
        if len(x) < 20:          # too few firms to estimate tail
            continue
        yearly[yr] = estimate_single_cross_section(x, sigma)

    if not yearly:
        return {"n_firms": 0, "xi": np.nan, "theta": np.nan,
                "sigma": sigma, "yearly": {}, "robustness": {}}

    # Average xi across years at each cutoff
    avg_robust = {}
    for key in [f"{int(f*100)}pct" for f in TAIL_CUTOFFS]:
        hill_vals = [v["robustness"][key]["hill"]
                     for v in yearly.values()
                     if v["robustness"][key]["hill"] is not None]
        ols_vals  = [v["robustness"][key]["ols"]
                     for v in yearly.values()
                     if v["robustness"][key]["ols"]  is not None]
        xi_vals   = [v["robustness"][key]["xi"]
                     for v in yearly.values()
                     if v["robustness"][key]["xi"]   is not None]
        avg_robust[key] = {
            "hill_mean": round(np.mean(hill_vals), 4) if hill_vals else None,
            "ols_mean":  round(np.mean(ols_vals),  4) if ols_vals  else None,
            "xi_mean":   round(np.mean(xi_vals),   4) if xi_vals   else None,
            "n_years":   len(xi_vals),
        }

    xi_primary = avg_robust[f"{int(PRIMARY_CUTOFF*100)}pct"]["xi_mean"]
    n_total    = sum(v["n_firms"] for v in yearly.values())

    # Apply floor: if xi < 1, theta would violate theta > sigma - 1.
    # Set theta = sigma (minimum well-defined value) and flag it.
    if xi_primary and xi_primary < 1.0 and APPLY_FLOOR:
        theta     = round(sigma, 4)
        floored   = True
    else:
        theta     = round(xi_primary * (sigma - 1), 4) if xi_primary else np.nan
        floored   = False

    return {
        "n_firms":      n_total,
        "n_years":      len(yearly),
        "years_used":   list(yearly.keys()),
        "sigma":        sigma,
        "xi":           xi_primary,
        "theta":        theta,
        "floored":      floored,
        "robustness":   avg_robust,
        "yearly_xi":    {yr: v["xi"] for yr, v in yearly.items()},
    }


# ── Hill plots ───────────────────────────────────────────────────────────────

def save_hill_plots(df: pd.DataFrame, sector_map: dict,
                    years_map: dict, grouping_name: str):
    """
    Save Hill plots for all sectors. For multi-year sectors, plot each year
    as a separate line. For T (single year), plot just that year.
    """
    n_sectors = len(sector_map)
    fig, axes = plt.subplots(1, n_sectors, figsize=(4 * n_sectors, 4))
    if n_sectors == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "green", "red"]
    for ax, (s, codes) in zip(axes, sector_map.items()):
        years = years_map.get(s, YEARS_O)
        for i, yr in enumerate(years):
            x = get_active_exports(df, codes, yr)
            if len(x) < 20:
                continue
            x_sorted = np.sort(x)[::-1]
            ks, xis  = hill_plot_data(x_sorted)
            ax.plot(ks, xis, lw=1.0, color=colors[i % len(colors)],
                    label=str(yr), alpha=0.8)
        ax.set_xlabel("k (tail firms)")
        ax.set_ylabel("Hill xi")
        ax.set_title(f"Sector {s}")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 4)

    fig.suptitle(f"Hill plots — {grouping_name}", y=1.02)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR,
                         f"hill_plot_{grouping_name.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    return fname


# ── main ─────────────────────────────────────────────────────────────────────

def make_years_map(sector_map: dict) -> dict:
    """T uses YEARS_T, all others use YEARS_O."""
    return {s: (YEARS_T if s == "T" else YEARS_O) for s in sector_map}


def estimate_grouping(df: pd.DataFrame, sector_map: dict) -> dict:
    results = {}
    years_map = make_years_map(sector_map)
    for s, codes in sector_map.items():
        results[s] = estimate_sector(df, codes, years_map[s], SIGMA[s])
    return results


def update_params_file(result: dict, out_path: str = OUT_PATH):
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            params = json.load(f)
    else:
        params = {}
    params["theta_s"] = result
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    return out_path


# ── reporting ────────────────────────────────────────────────────────────────

def print_grouping(name: str, g: dict):
    sep = "-" * 82
    print(f"\n  {sep}")
    print(f"  {name}")
    print(f"  {sep}")
    print(f"  {'Sector':<6}  {'Label':<42}  {'N firms':>8}  {'yrs':>4}  "
          f"{'sigma':>6}  {'xi(10%)':>8}  {'theta':>8}  {'OK?':>5}")
    print(f"  {sep}")
    for s, d in g.items():
        lbl     = SECTOR_LABELS.get(s, s)
        check   = "OK" if (d["theta"] and d["theta"] > (d["sigma"] - 1)) else "FAIL"
        floored = " [floor]" if d.get("floored") else ""
        xi_s    = f"{d['xi']:.4f}"    if d["xi"]    else "  N/A"
        th_s    = f"{d['theta']:.4f}" if d["theta"] else "  N/A"
        print(f"  {s:<6}  {lbl:<42}  {d['n_firms']:>8,}  "
              f"{d['n_years']:>4}  {d['sigma']:>6.1f}  "
              f"{xi_s:>8}  {th_s:>8}  {check:>5}{floored}")
    print(f"  {sep}")

    print(f"\n  Yearly xi breakdown (10% tail):")
    print(f"  {'Sector':<6}  {'2005':>8}  {'2006':>8}  {'2007':>8}  {'2008':>8}  {'mean':>8}")
    print(f"  {'-'*50}")
    for s, d in g.items():
        yxi = d.get("yearly_xi", {})
        vals = [f"{yxi.get(yr, float('nan')):8.4f}" for yr in [2005,2006,2007,2008]]
        print(f"  {s:<6}  {'  '.join(vals)}  {d['xi']:8.4f}" if d["xi"] else
              f"  {s:<6}  {'  '.join(vals)}     N/A")

    print(f"\n  Robustness (avg xi across years, by tail cutoff):")
    print(f"  {'Sector':<6}  {'5% Hill':>9}  {'5% OLS':>9}  "
          f"{'10% Hill':>9}  {'10% OLS':>9}  {'20% Hill':>9}  {'20% OLS':>9}")
    print(f"  {'-'*68}")
    for s, d in g.items():
        r = d["robustness"]
        def fmt(v): return f"{v:9.4f}" if v is not None else "      N/A"
        print(f"  {s:<6}  "
              f"{fmt(r['5pct']['hill_mean'])}  {fmt(r['5pct']['ols_mean'])}  "
              f"{fmt(r['10pct']['hill_mean'])}  {fmt(r['10pct']['ols_mean'])}  "
              f"{fmt(r['20pct']['hill_mean'])}  {fmt(r['20pct']['ols_mean'])}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  Total observations: {len(df):,}")

    print("Estimating xi year-by-year, averaging across years...")
    print(f"  T sector: {YEARS_T}")
    print(f"  Other sectors: {YEARS_O}")

    g1 = estimate_grouping(df, SECTORS_G1)
    g2 = estimate_grouping(df, SECTORS_G2)

    # Hill plots
    p1 = save_hill_plots(df, SECTORS_G1, make_years_map(SECTORS_G1), "grouping_1")
    p2 = save_hill_plots(df, SECTORS_G2, make_years_map(SECTORS_G2), "grouping_2")

    result = {
        "grouping_1": {
            "description": "Baseline: T + O",
            "sectors":     list(g1.keys()),
            "theta":       {s: v["theta"] for s, v in g1.items()},
            "xi":          {s: v["xi"]    for s, v in g1.items()},
            "details":     g1,
        },
        "grouping_2": {
            "description": "Extended: T + S1 + S2 + S3 + S4 + O",
            "sectors":     list(g2.keys()),
            "theta":       {s: v["theta"] for s, v in g2.items()},
            "xi":          {s: v["xi"]    for s, v in g2.items()},
            "details":     g2,
        },
        "sector_labels": SECTOR_LABELS,
        "sigma_used":    SIGMA,
        "source":        "Egyptian customs data, pre-period 2005-2008, "
                         "collapsed to firm averages over active years",
        "method":        ("xi estimated via Hill (corrected, x_(k+1) threshold) "
                          "and OLS log-rank with Blom correction (rank-0.5), "
                          "averaged at 10% tail cutoff. "
                          "theta = xi * (sigma - 1)."),
        "note":          "theta > sigma - 1 required for finite mean firm sales.",
    }

    print("\n" + "=" * 82)
    print("  theta_s: Pareto shape parameter")
    print("=" * 82)
    print_grouping("Grouping 1 - Baseline", g1)
    print_grouping("Grouping 2 - Extended", g2)

    saved = update_params_file(result)
    print(f"\n  Saved to: {saved}")
    print(f"  Hill plots: {p1}")
    print(f"             {p2}")
