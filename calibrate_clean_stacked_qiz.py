"""
Clean one-region QIZ calibration.

This script implements the economic specification we want to test:

- QIZ compliance does not replace the normal US export fixed cost.
- QIZ compliance adds an ROO/compliance cost on top of normal exporting.
- Israeli input costs enter only US-destined QIZ production.
- The Israeli input premium is fixed at p_IL / p_RW = 1.20.
- lambda_RW and pre-policy fixed costs use the best one-region pre-policy fit
  already obtained from the continuous-lambda calibration.

The point is diagnostic as much as calibrating: if this clean specification
cannot match textile compliance among US exporters, the old route-specific
fixed-cost model was fitting the target through the wrong margin.
"""

from __future__ import annotations

import csv
import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pyreadstat

import qiz_model_ge as qm


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "clean_stacked_qiz_calibration.json")
OUT_CSV = os.path.join(ROOT, "clean_stacked_qiz_grid.csv")
OUT_MD = os.path.join(ROOT, "clean_stacked_qiz_results.md")

DATA_PATH = (
    r"C:\Users\Admin\Desktop\Idea QIZs and Development"
    r"\Export and Import Data Egypt\final_data_matched.dta"
)

REGION = "Q"
SECTORS = ["T", "O"]

PREPOLICY_TARGETS = {
    "T": {"dom_only": 0.8927, "exp_US": 0.0575, "exp_RW": 0.0690},
    "O": {"dom_only": 0.8492, "exp_US": 0.0112, "exp_RW": 0.1425},
}

# Reliable one-region off-state fit from the continuous-lambda stage-1 run.
STAGE1_PARAMS = {
    "f_dom": {"T": 2.1458107843752035, "O": 1.9659963834167784},
    "f_export_US": {"T": 51.92150388783668, "O": 10.818774524863166},
    "f_export_RW": {"T": 10.349215443421434, "O": 1.3879291617442884},
    "lambda_RW": 0.18996000235111887,
}


def clean_float(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_float(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return x


def compute_empirical_targets() -> Dict[str, Any]:
    cols = [
        "Year",
        "Trader_ID",
        "is_textile",
        "exp_from_us",
        "exp_from_nonus",
        "imp_from_israel",
        "treatment_year",
    ]
    df, _ = pyreadstat.read_dta(DATA_PATH, usecols=cols)
    df = df[(df["Year"] >= 2005) & (df["Year"] <= 2016)].copy()
    for c in ["exp_from_us", "exp_from_nonus", "imp_from_israel"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["exports_us"] = df["exp_from_us"] > 0
    df["imports_isr"] = df["imp_from_israel"] > 0

    out: Dict[str, Any] = {}
    for s, is_textile in [("T", True), ("O", False)]:
        sub = df[df["is_textile"].eq(1) if is_textile else df["is_textile"].ne(1)]
        firm = sub.groupby("Trader_ID").agg(
            ever_us=("exports_us", "max"),
            ever_isr=("imports_isr", "max"),
        )
        denom = int(firm["ever_us"].sum())
        num = int((firm["ever_us"] & firm["ever_isr"]).sum())
        all_firms = int(len(firm))
        ever_isr_all = int(firm["ever_isr"].sum())
        out[s] = {
            "conditional_on_ever_us": {
                "denom": denom,
                "num": num,
                "rate": float(num / denom) if denom else None,
            },
            "all_customs_observed": {
                "denom": all_firms,
                "num": ever_isr_all,
                "rate": float(ever_isr_all / all_firms) if all_firms else None,
            },
        }

    # Mechanism target: non-US export growth among textile firms that ever import
    # from Israel, around first Israeli import year.
    t = df[df["is_textile"].eq(1)].copy()
    ever_isr = t.groupby("Trader_ID")["imports_isr"].max()
    t = t[t["Trader_ID"].isin(ever_isr.index[ever_isr])].copy()
    treatment_year = (
        t.dropna(subset=["treatment_year"])
        .groupby("Trader_ID")["treatment_year"]
        .first()
    )
    rows: List[Dict[str, float]] = []
    for trader_id, year0 in treatment_year.items():
        sub = t[t["Trader_ID"].eq(trader_id)]
        pre = sub.loc[sub["Year"].eq(year0 - 1), "exp_from_nonus"]
        post = sub.loc[sub["Year"].eq(year0 + 1), "exp_from_nonus"]
        if len(pre) == 0 or len(post) == 0:
            continue
        rows.append(
            {
                "pre": float(pre.max()),
                "post": float(post.max()),
            }
        )
    chg = pd.DataFrame(rows)
    pre_sum = float(chg["pre"].sum())
    post_sum = float(chg["post"].sum())
    positive_pre = chg[chg["pre"] > 0].copy()
    out["T_nonus_growth_among_israel_importers"] = {
        "n": int(len(chg)),
        "positive_growth_share": float((chg["post"] > chg["pre"]).mean()),
        "aggregate_pre": pre_sum,
        "aggregate_post": post_sum,
        "aggregate_pct": float(100.0 * (post_sum / max(pre_sum, 1.0e-12) - 1.0)),
        "median_pct_positive_pre": (
            float(100.0 * ((positive_pre["post"] / positive_pre["pre"]).median() - 1.0))
            if len(positive_pre)
            else None
        ),
    }
    return out


def build_params(n_phi: int = 80, n_eps: int = 5, intensity_grid: int = 5) -> Dict[str, Any]:
    p = qm.params_defensible()
    qm.restrict_to_regions(p, [REGION])

    p["lambda_RW"] = STAGE1_PARAMS["lambda_RW"]
    for s in SECTORS:
        p["f_dom"][(REGION, s)] = STAGE1_PARAMS["f_dom"][s]
        p["f_export"][(REGION, "US", s)] = STAGE1_PARAMS["f_export_US"][s]
        p["f_export"][(REGION, "RW", s)] = STAGE1_PARAMS["f_export_RW"][s]

    p["roo_cost_formula"] = "normalized"
    p["roo_cost_scope"] = "US_only"
    p["qiz_us_fixed_cost_mode"] = "stacked"
    p["p_il"]["T"] = 1.20
    p["p_il"]["O"] = 1.20

    p["upgrade_mode"] = "continuous"
    p["upgrade_requires_US"] = True
    p["upgrade_psi"]["T"] = 0.0
    p["upgrade_psi"]["O"] = 0.0
    p["upgrade_psi_comp"]["O"] = 0.0
    p["upgrade_cost_fixed"]["T"] = 0.0
    p["upgrade_cost_lin"]["T"] = 0.0
    p["upgrade_intensity_max"]["T"] = 2.0
    p["upgrade_intensity_max"]["O"] = 2.0
    p["upgrade_intensity_grid_size"] = int(intensity_grid)

    p["sigma_C"]["T"] = 1.2
    p["sigma_C"]["O"] = 0.8
    p["fC_mean"]["O"] = 1000.0

    p["f_entry"][(REGION, "T")] = 1.0
    p["f_entry"][(REGION, "O")] = 1.0

    p["n_phi"] = int(n_phi)
    p["n_eps"] = int(n_eps)
    p["goods_max_iter"] = 900
    p["outer_max_iter"] = 700
    p["outer_step"] = 0.15
    p["goods_cycle_tol"] = 0.04
    p["outer_cycle_tol"] = 0.06
    p["pair_refine_max_iter"] = 2
    p["pair_refine_tol"] = 1.0e-5
    p["outer_tol"] = 1.0e-5
    return p


def participation_moments(sol: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for s in SECTORS:
        m = sol["moments"][(REGION, s)]
        out[s] = {
            "dom_only": float(m["domestic_only_share_among_active"]),
            "exp_US": float(m["US_export_share_among_active"]),
            "exp_RW": float(m["RW_export_share_among_active"]),
            "active_share": float(m["active_share"]),
        }
    return out


def percent_change(on: float, off: float) -> float:
    return float(100.0 * (on / max(abs(off), 1.0e-12) - 1.0))


def evaluate(
    p: Dict[str, Any],
    fixed_off: Dict[str, Any] | None = None,
    off_seed: Dict[str, Any] | None = None,
    on_seed: Dict[str, Any] | None = None,
    disable_upgrade: bool = False,
) -> Dict[str, Any]:
    if fixed_off is None:
        off = qm.solve_equilibrium(
            deepcopy(p),
            qiz_on=False,
            disable_upgrade=True,
            initial_state=off_seed,
            verbose=False,
        )
    else:
        off = fixed_off
    on = qm.solve_equilibrium(
        deepcopy(p),
        qiz_on=True,
        disable_upgrade=disable_upgrade,
        initial_state=on_seed or off,
        verbose=False,
    )
    same = qm.summarize_qt_same_type_changes(on, off, p, disable_upgrade=disable_upgrade)
    trade_off = qm.summarize_trade(off, p)
    trade_on = qm.summarize_trade(on, p)
    sector_off = qm.summarize_trade_by_sector(off, p)
    sector_on = qm.summarize_trade_by_sector(on, p)

    us_off = float(trade_off["exports"][(REGION, "US")])
    us_on = float(trade_on["exports"][(REGION, "US")])
    rw_off = float(trade_off["exports"][(REGION, "RW")])
    rw_on = float(trade_on["exports"][(REGION, "RW")])
    t_rw_off = float(sector_off[(REGION, "T", "RW")])
    t_rw_on = float(sector_on[(REGION, "T", "RW")])
    t_mom_on = on["moments"][(REGION, "T")]
    t_comp_active = float(t_mom_on["compliance_share_among_active"])
    t_upgrade_intensity_active = float(t_mom_on["upgrade_intensity_among_active"])
    t_upgrade_intensity_per_complier = (
        t_upgrade_intensity_active / max(t_comp_active, 1.0e-12)
        if t_comp_active > 0
        else 0.0
    )

    return {
        "prepolicy_fit": participation_moments(off),
        "T_comp_US": float(t_mom_on["compliance_share_among_US_exporters"]),
        "T_comp_active": t_comp_active,
        "T_US_active": float(t_mom_on["US_export_share_among_active"]),
        "T_RW_active": float(t_mom_on["RW_export_share_among_active"]),
        "O_comp_US": float(on["moments"][(REGION, "O")]["compliance_share_among_US_exporters"]),
        "O_US_active": float(on["moments"][(REGION, "O")]["US_export_share_among_active"]),
        "comp_rw_pct": float(same["comp"]["RW_pct"]) if same["comp"]["defined"] else None,
        "comp_us_pct": float(same["comp"]["US_pct"]) if same["comp"]["defined"] else None,
        "comp_total_pct": float(same["comp"]["TOT_pct"]) if same["comp"]["defined"] else None,
        "T_sector_RW_pct": percent_change(t_rw_on, t_rw_off) if t_rw_off > 0 else None,
        "T_upgrade_intensity_active": t_upgrade_intensity_active,
        "comp_avg_upgrade_intensity": t_upgrade_intensity_per_complier,
        "welfare_pct": percent_change(float(on["welfare"]), float(off["welfare"])),
        "exports": {
            "US_off": us_off,
            "US_on": us_on,
            "US_pct": percent_change(us_on, us_off) if us_off > 0 else None,
            "RW_off": rw_off,
            "RW_on": rw_on,
            "RW_pct": percent_change(rw_on, rw_off) if rw_off > 0 else None,
        },
    }


def loss(row: Dict[str, Any], targets: Dict[str, Any]) -> float:
    t_comp = targets["T"]["conditional_on_ever_us"]["rate"]
    nonus_target = targets["T_nonus_growth_among_israel_importers"]["aggregate_pct"]
    terms = []
    terms.append(((row["T_comp_US"] - t_comp) / 0.10) ** 2)
    # Tie-breaker only. This is not the empirical target, but it highlights cases
    # where the model gets the wrong conditional rate while matching active take-up.
    terms.append(0.01 * ((row["T_comp_active"] - t_comp) / 0.10) ** 2)
    # Treat O take-up as a validation/penalty. The model target is close to zero
    # because true QIZ use outside textiles should be low under stacked costs.
    terms.append((row["O_comp_US"] / 0.05) ** 2)
    rw_model = row["comp_rw_pct"] if row["comp_rw_pct"] is not None else row.get("T_sector_RW_pct")
    if rw_model is not None and np.isfinite(rw_model):
        terms.append(((rw_model - nonus_target) / 75.0) ** 2)
    else:
        terms.append(100.0)
    return float(sum(terms))


def set_policy_params(
    base: Dict[str, Any],
    fC_T: float,
    psi_comp_T: float,
    upgrade_cost_quad_T: float,
) -> Dict[str, Any]:
    p = deepcopy(base)
    p["fC_mean"]["T"] = float(fC_T)
    p["upgrade_psi_comp"]["T"] = float(psi_comp_T)
    p["upgrade_cost_quad"]["T"] = float(upgrade_cost_quad_T)
    return p


def run_grid(targets: Dict[str, Any]) -> Dict[str, Any]:
    base = build_params()
    off_base = qm.solve_equilibrium(
        deepcopy(base),
        qiz_on=False,
        disable_upgrade=True,
        initial_state=None,
        verbose=False,
    )

    tariff_rows: List[Dict[str, Any]] = []
    for fC_T in [0.0, 0.1, 1.0, 10.0, 100.0]:
        p = set_policy_params(base, fC_T=fC_T, psi_comp_T=0.0, upgrade_cost_quad_T=100.0)
        try:
            ev = evaluate(p, fixed_off=off_base, disable_upgrade=True)
            tariff_rows.append({"case": "tariffs_only", "fC_T": fC_T, **ev})
        except RuntimeError as exc:
            tariff_rows.append({"case": "tariffs_only", "fC_T": fC_T, "error": str(exc)})

    rows: List[Dict[str, Any]] = []
    for psi_comp_T in [0.10, 0.15, 0.25, 0.50]:
        for upgrade_cost_quad_T in [0.0, 0.8]:
            for fC_T in [0.0, 10.0, 20.0, 80.0, 320.0]:
                p = set_policy_params(
                    base,
                    fC_T=fC_T,
                    psi_comp_T=psi_comp_T,
                    upgrade_cost_quad_T=upgrade_cost_quad_T,
                )
                try:
                    ev = evaluate(p, fixed_off=off_base, disable_upgrade=False)
                    row = {
                        "case": "stacked_continuous_upgrade",
                        "fC_T": fC_T,
                        "psi_comp_T": psi_comp_T,
                        "upgrade_cost_quad_T": upgrade_cost_quad_T,
                        **ev,
                    }
                    row["loss"] = loss(row, targets)
                    rows.append(row)
                except RuntimeError as exc:
                    rows.append(
                        {
                            "case": "stacked_continuous_upgrade",
                            "fC_T": fC_T,
                            "psi_comp_T": psi_comp_T,
                            "upgrade_cost_quad_T": upgrade_cost_quad_T,
                            "error": str(exc),
                            "loss": float("inf"),
                        }
                    )

    valid = [r for r in rows if "error" not in r]
    best = min(valid, key=lambda r: r["loss"]) if valid else None
    t_comp = targets["T"]["conditional_on_ever_us"]["rate"]
    closest_active = min(valid, key=lambda r: abs(r["T_comp_active"] - t_comp)) if valid else None
    return {
        "targets": targets,
        "base_specification": {
            "regions": [REGION],
            "qiz_us_fixed_cost_mode": "stacked",
            "roo_cost_formula": "normalized",
            "roo_cost_scope": "US_only",
            "p_il_over_p_rw": 1.20,
            "lambda_RW": STAGE1_PARAMS["lambda_RW"],
            "upgrade_mode": "continuous",
            "upgrade_requires_US": True,
            "fC_O": base["fC_mean"]["O"],
            "sigma_C": {s: base["sigma_C"][s] for s in SECTORS},
            "grids": {
                "n_phi": base["n_phi"],
                "n_eps": base["n_eps"],
                "upgrade_intensity_grid_size": base["upgrade_intensity_grid_size"],
            },
        },
        "tariffs_only": tariff_rows,
        "grid_rows": rows,
        "best": best,
        "closest_active_share": closest_active,
    }


def write_grid_csv(rows: Iterable[Dict[str, Any]]) -> None:
    cols = [
        "case",
        "fC_T",
        "psi_comp_T",
        "upgrade_cost_quad_T",
        "loss",
        "T_comp_US",
        "T_comp_active",
        "T_US_active",
        "T_RW_active",
        "O_comp_US",
        "O_US_active",
        "comp_rw_pct",
        "T_sector_RW_pct",
        "comp_us_pct",
        "comp_total_pct",
        "comp_avg_upgrade_intensity",
        "welfare_pct",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(summary: Dict[str, Any]) -> None:
    best = summary["best"]
    closest_active = summary.get("closest_active_share")
    targets = summary["targets"]

    def fmt_pct_or_na(x: Any) -> str:
        return "N/A" if x is None else f"{float(x):.1f}%"

    with open(OUT_MD, "w") as f:
        f.write("# Clean Stacked QIZ Calibration\n\n")
        f.write("## Specification\n\n")
        f.write("- One Egypt region: `Q`\n")
        f.write("- `qiz_us_fixed_cost_mode = stacked`\n")
        f.write("- `roo_cost_formula = normalized`\n")
        f.write("- Israeli input premium: `p_IL / p_RW = 1.20`\n")
        f.write("- `lambda_RW = 0.18996`\n")
        f.write("- Continuous upgrading with QIZ complementarity\n\n")
        f.write("## Empirical Targets\n\n")
        f.write(
            f"- Textile compliance among ever-US exporters: "
            f"{targets['T']['conditional_on_ever_us']['rate']:.4f} "
            f"({targets['T']['conditional_on_ever_us']['num']}/"
            f"{targets['T']['conditional_on_ever_us']['denom']})\n"
        )
        f.write(
            f"- Non-textile Israel-import proxy among all customs-observed firms: "
            f"{targets['O']['all_customs_observed']['rate']:.4f}\n"
        )
        f.write(
            f"- Textile non-US export growth around first Israeli import: "
            f"{targets['T_nonus_growth_among_israel_importers']['aggregate_pct']:.1f}% "
            f"aggregate, "
            f"{targets['T_nonus_growth_among_israel_importers']['median_pct_positive_pre']:.1f}% "
            f"median among firms with positive pre exports\n\n"
        )
        f.write("## Tariffs-Only Diagnostic\n\n")
        for row in summary["tariffs_only"]:
            if "error" in row:
                f.write(f"- `fC_T={row['fC_T']}` failed: {row['error']}\n")
            else:
                f.write(
                    f"- `fC_T={row['fC_T']}`: textile compliance among US exporters "
                    f"`{row['T_comp_US']:.4f}`, textile US exporter share "
                    f"`{row['T_US_active']:.4f}`\n"
                )
        f.write("\n## Best Grid Point\n\n")
        if best is None:
            f.write("No stable grid point found.\n")
            return
        f.write(f"- `fC_T = {best['fC_T']}`\n")
        f.write(f"- `upgrade_psi_comp_T = {best['psi_comp_T']}`\n")
        f.write(f"- `upgrade_cost_quad_T = {best['upgrade_cost_quad_T']}`\n")
        f.write(f"- Textile compliance among US exporters: `{best['T_comp_US']:.4f}`\n")
        f.write(f"- Textile compliance among active firms: `{best['T_comp_active']:.4f}`\n")
        f.write(f"- Non-textile compliance among US exporters: `{best['O_comp_US']:.4f}`\n")
        f.write(f"- Complier non-US export change: `{fmt_pct_or_na(best['comp_rw_pct'])}`\n")
        f.write(f"- Textile-sector non-US export change: `{fmt_pct_or_na(best['T_sector_RW_pct'])}`\n")
        f.write(f"- Upgrade intensity per complier: `{best['comp_avg_upgrade_intensity']:.3f}`\n")
        f.write(f"- Welfare change: `{best['welfare_pct']:.3f}%`\n\n")
        if closest_active is not None:
            f.write("## Closest Active-Share Point\n\n")
            f.write(
                "This point gets the active-firm compliance share close to the "
                "empirical 32% number, but it still fails the correct conditional "
                "moment because all US exporters comply.\n\n"
            )
            f.write(f"- `fC_T = {closest_active['fC_T']}`\n")
            f.write(f"- `upgrade_psi_comp_T = {closest_active['psi_comp_T']}`\n")
            f.write(f"- Textile compliance among active firms: `{closest_active['T_comp_active']:.4f}`\n")
            f.write(f"- Textile compliance among US exporters: `{closest_active['T_comp_US']:.4f}`\n\n")
        f.write("## Interpretation\n\n")
        f.write(
            "Under stacked costs, the model no longer creates non-textile QIZ use, "
            "which is the intended behavior. The remaining failure is textile take-up: "
            "whenever the policy generates textile US exporters, those exporters are "
            "almost always QIZ compliers. The model can move the active-firm compliance "
            "share around, but it does not produce an interior conditional rate among "
            "US exporters. To match the 32% target, the model needs a real incumbent "
            "MFN exporter margin, richer compliance heterogeneity, or an entry structure "
            "that separates QIZ entrants from incumbent MFN exporters.\n"
        )


def main() -> None:
    targets = compute_empirical_targets()
    summary = run_grid(targets)
    with open(OUT_JSON, "w") as f:
        json.dump(clean_float(summary), f, indent=2)
    write_grid_csv(summary["grid_rows"])
    write_markdown(clean_float(summary))
    best = summary["best"]
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_MD}")
    if best is not None:
        comp_rw_print = "N/A" if best["comp_rw_pct"] is None else f"{best['comp_rw_pct']:.1f}"
        print(
            "Best:",
            f"fC_T={best['fC_T']}",
            f"psi_comp_T={best['psi_comp_T']}",
            f"upgrade_cost_quad_T={best['upgrade_cost_quad_T']}",
            f"T_comp_US={best['T_comp_US']:.4f}",
            f"O_comp_US={best['O_comp_US']:.4f}",
            f"comp_rw_pct={comp_rw_print}",
            f"T_sector_RW_pct={best['T_sector_RW_pct']:.1f}",
        )


if __name__ == "__main__":
    main()
