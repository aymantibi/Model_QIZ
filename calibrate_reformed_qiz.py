"""Reformed one-region QIZ calibration.

This specification fixes two problems in the clean stacked diagnostic:

1. Upgrading is not mechanically shut down without QIZ. All firms can choose
   continuous upgrading; QIZ compliance can add a complementarity, but it is
   not the only source of productivity upgrading.
2. Textile US exporters have a real MFN-vs-QIZ margin. A latent eps type both
   raises QIZ compliance costs and lowers MFN US continuation costs, capturing
   incumbent US relationships that are costly to switch into QIZ.
"""

from __future__ import annotations

import csv
import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import qiz_model_ge as qm
from calibrate_clean_stacked_qiz import (
    PREPOLICY_TARGETS,
    REGION,
    SECTORS,
    STAGE1_PARAMS,
    clean_float,
    compute_empirical_targets,
    participation_moments,
    percent_change,
)


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "reformed_qiz_calibration.json")
OUT_CSV = os.path.join(ROOT, "reformed_qiz_grid.csv")
OUT_MD = os.path.join(ROOT, "reformed_qiz_results.md")


def build_params(n_phi: int = 70, n_eps: int = 7, intensity_grid: int = 5) -> Dict[str, Any]:
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
    p["upgrade_requires_US"] = False
    p["upgrade_psi"]["O"] = 0.02
    p["upgrade_psi_comp"]["O"] = 0.0
    p["upgrade_cost_fixed"]["T"] = 0.0
    p["upgrade_cost_lin"]["T"] = 0.0
    p["upgrade_intensity_max"]["T"] = 2.0
    p["upgrade_intensity_max"]["O"] = 2.0
    p["upgrade_intensity_grid_size"] = int(intensity_grid)

    p["sigma_C"]["O"] = 0.8
    p["fC_mean"]["O"] = 1000.0
    p["mfn_us_fixed_cost_discount_sigma"]["O"] = 0.0

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


def set_policy_params(
    base: Dict[str, Any],
    fC_T: float,
    sigma_C_T: float,
    mfn_incumbency_T: float,
    upgrade_psi_T: float,
    upgrade_psi_comp_T: float,
    upgrade_cost_quad_T: float,
    upgrade_cost_comp_mult_T: float,
) -> Dict[str, Any]:
    p = deepcopy(base)
    p["fC_mean"]["T"] = float(fC_T)
    p["sigma_C"]["T"] = float(sigma_C_T)
    p["mfn_us_fixed_cost_discount_sigma"]["T"] = float(mfn_incumbency_T)
    p["mfn_us_fixed_cost_min_mult"]["T"] = 0.15
    p["mfn_us_fixed_cost_max_mult"]["T"] = 6.0
    p["upgrade_psi"]["T"] = float(upgrade_psi_T)
    p["upgrade_psi_comp"]["T"] = float(upgrade_psi_comp_T)
    p["upgrade_cost_quad"]["T"] = float(upgrade_cost_quad_T)
    p["upgrade_cost_comp_mult"]["T"] = float(upgrade_cost_comp_mult_T)
    return p


def solve_off(p: Dict[str, Any], seed: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return qm.solve_equilibrium(
        deepcopy(p),
        qiz_on=False,
        disable_upgrade=False,
        initial_state=seed,
        verbose=False,
    )


def evaluate(
    p: Dict[str, Any],
    fixed_off: Dict[str, Any] | None = None,
    off_seed: Dict[str, Any] | None = None,
    on_seed: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    off = fixed_off if fixed_off is not None else solve_off(p, seed=off_seed)
    p_on = deepcopy(p)
    on = qm.solve_equilibrium(
        p_on,
        qiz_on=True,
        disable_upgrade=False,
        initial_state=on_seed or off,
        verbose=False,
    )

    same = qm.summarize_qt_same_type_changes(on, off, p_on, disable_upgrade=False)
    groups = qm.summarize_qt_groups(on, p_on, qiz_on=True, disable_upgrade=False)
    trade_off = qm.summarize_trade(off, p)
    trade_on = qm.summarize_trade(on, p_on)
    sector_off = qm.summarize_trade_by_sector(off, p)
    sector_on = qm.summarize_trade_by_sector(on, p_on)

    t_mom_off = off["moments"][(REGION, "T")]
    t_mom_on = on["moments"][(REGION, "T")]
    o_mom_on = on["moments"][(REGION, "O")]
    t_comp_active = float(t_mom_on["compliance_share_among_active"])
    t_us_active = float(t_mom_on["US_export_share_among_active"])
    t_mfn_us_active = max(0.0, t_us_active - t_comp_active)

    us_off = float(trade_off["exports"][(REGION, "US")])
    us_on = float(trade_on["exports"][(REGION, "US")])
    rw_off = float(trade_off["exports"][(REGION, "RW")])
    rw_on = float(trade_on["exports"][(REGION, "RW")])
    t_rw_off = float(sector_off[(REGION, "T", "RW")])
    t_rw_on = float(sector_on[(REGION, "T", "RW")])

    return {
        "prepolicy_fit": participation_moments(off),
        "T_comp_US": float(t_mom_on["compliance_share_among_US_exporters"]),
        "T_comp_active": t_comp_active,
        "T_MFN_US_active": t_mfn_us_active,
        "T_US_active": t_us_active,
        "T_RW_active": float(t_mom_on["RW_export_share_among_active"]),
        "O_comp_US": float(o_mom_on["compliance_share_among_US_exporters"]),
        "O_US_active": float(o_mom_on["US_export_share_among_active"]),
        "T_upgrade_intensity_off_active": float(t_mom_off["upgrade_intensity_among_active"]),
        "T_upgrade_intensity_on_active": float(t_mom_on["upgrade_intensity_among_active"]),
        "T_upgrade_intensity_change": float(
            t_mom_on["upgrade_intensity_among_active"]
            - t_mom_off["upgrade_intensity_among_active"]
        ),
        "comp_avg_upgrade_intensity": float(groups["comp"]["avg_upgrade_intensity"]),
        "noncomp_avg_upgrade_intensity": float(groups["noncomp"]["avg_upgrade_intensity"]),
        "upgrade_intensity_gap_comp_noncomp": float(
            groups["comp"]["avg_upgrade_intensity"]
            - groups["noncomp"]["avg_upgrade_intensity"]
        ),
        "comp_rw_pct": float(same["comp"]["RW_pct"]) if same["comp"]["defined"] else None,
        "comp_us_pct": float(same["comp"]["US_pct"]) if same["comp"]["defined"] else None,
        "comp_total_pct": float(same["comp"]["TOT_pct"]) if same["comp"]["defined"] else None,
        "T_sector_RW_pct": percent_change(t_rw_on, t_rw_off) if t_rw_off > 0 else None,
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


def prepolicy_loss(pre: Dict[str, Dict[str, float]]) -> float:
    terms = []
    for s in SECTORS:
        target = PREPOLICY_TARGETS[s]
        for k in ["dom_only", "exp_US", "exp_RW"]:
            terms.append(((pre[s][k] - target[k]) / 0.04) ** 2)
    return float(0.10 * sum(terms))


def loss(row: Dict[str, Any], targets: Dict[str, Any]) -> float:
    t_comp = targets["T"]["conditional_on_ever_us"]["rate"]
    nonus_target = targets["T_nonus_growth_among_israel_importers"]["aggregate_pct"]
    terms = []
    terms.append(((row["T_comp_US"] - t_comp) / 0.08) ** 2)
    terms.append((row["O_comp_US"] / 0.04) ** 2)
    terms.append(prepolicy_loss(row["prepolicy_fit"]))
    rw_model = row["comp_rw_pct"] if row["comp_rw_pct"] is not None else row.get("T_sector_RW_pct")
    if rw_model is not None and np.isfinite(rw_model):
        terms.append(((rw_model - nonus_target) / 90.0) ** 2)
    else:
        terms.append(25.0)
    # Keep the intended mechanism: QIZ should increase upgrading, but upgrading
    # should not be absent in the no-QIZ economy.
    terms.append((max(0.0, 0.05 - row["T_upgrade_intensity_off_active"]) / 0.05) ** 2)
    terms.append((max(0.0, 0.02 - row["T_upgrade_intensity_change"]) / 0.04) ** 2)
    terms.append((max(0.0, 0.25 - row["comp_avg_upgrade_intensity"]) / 0.15) ** 2)
    terms.append((max(0.0, 0.10 - row["upgrade_intensity_gap_comp_noncomp"]) / 0.15) ** 2)
    return float(sum(terms))


def run_grid(targets: Dict[str, Any]) -> Dict[str, Any]:
    base = build_params()
    off_cache: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    off_seed: Dict[str, Any] | None = None
    rows: List[Dict[str, Any]] = []

    grid = {
        "mfn_incumbency_T": [1.5, 2.0],
        "sigma_C_T": [1.2],
        "fC_T": [20.0, 40.0, 80.0],
        "upgrade_psi_T": [0.03],
        "upgrade_psi_comp_T": [0.02, 0.05],
        "upgrade_cost_quad_T": [2.0, 4.0],
        "upgrade_cost_comp_mult_T": [1.0],
    }

    for mfn_incumbency_T in grid["mfn_incumbency_T"]:
        for upgrade_psi_T in grid["upgrade_psi_T"]:
            for upgrade_cost_quad_T in grid["upgrade_cost_quad_T"]:
                off_key = (mfn_incumbency_T, upgrade_psi_T, upgrade_cost_quad_T)
                p_off = set_policy_params(
                    base,
                    fC_T=80.0,
                    sigma_C_T=1.2,
                    mfn_incumbency_T=mfn_incumbency_T,
                    upgrade_psi_T=upgrade_psi_T,
                    upgrade_psi_comp_T=0.0,
                    upgrade_cost_quad_T=upgrade_cost_quad_T,
                    upgrade_cost_comp_mult_T=1.0,
                )
                try:
                    off_cache[off_key] = solve_off(p_off, seed=off_seed)
                    off_seed = off_cache[off_key]
                except RuntimeError as exc:
                    for sigma_C_T in grid["sigma_C_T"]:
                        for fC_T in grid["fC_T"]:
                            for upgrade_psi_comp_T in grid["upgrade_psi_comp_T"]:
                                for upgrade_cost_comp_mult_T in grid["upgrade_cost_comp_mult_T"]:
                                    rows.append(
                                        {
                                            "mfn_incumbency_T": mfn_incumbency_T,
                                            "sigma_C_T": sigma_C_T,
                                            "fC_T": fC_T,
                                            "upgrade_psi_T": upgrade_psi_T,
                                            "upgrade_psi_comp_T": upgrade_psi_comp_T,
                                            "upgrade_cost_quad_T": upgrade_cost_quad_T,
                                            "upgrade_cost_comp_mult_T": upgrade_cost_comp_mult_T,
                                            "error": f"off failed: {exc}",
                                            "loss": float("inf"),
                                        }
                                    )
                    continue

                for sigma_C_T in grid["sigma_C_T"]:
                    for fC_T in grid["fC_T"]:
                        for upgrade_psi_comp_T in grid["upgrade_psi_comp_T"]:
                            for upgrade_cost_comp_mult_T in grid["upgrade_cost_comp_mult_T"]:
                                p = set_policy_params(
                                    base,
                                    fC_T=fC_T,
                                    sigma_C_T=sigma_C_T,
                                    mfn_incumbency_T=mfn_incumbency_T,
                                    upgrade_psi_T=upgrade_psi_T,
                                    upgrade_psi_comp_T=upgrade_psi_comp_T,
                                    upgrade_cost_quad_T=upgrade_cost_quad_T,
                                    upgrade_cost_comp_mult_T=upgrade_cost_comp_mult_T,
                                )
                                try:
                                    ev = evaluate(p, fixed_off=off_cache[off_key])
                                    row = {
                                        "case": "reformed_upgrade_mfn_margin",
                                        "mfn_incumbency_T": mfn_incumbency_T,
                                        "sigma_C_T": sigma_C_T,
                                        "fC_T": fC_T,
                                        "upgrade_psi_T": upgrade_psi_T,
                                        "upgrade_psi_comp_T": upgrade_psi_comp_T,
                                        "upgrade_cost_quad_T": upgrade_cost_quad_T,
                                        "upgrade_cost_comp_mult_T": upgrade_cost_comp_mult_T,
                                        **ev,
                                    }
                                    row["loss"] = loss(row, targets)
                                    rows.append(row)
                                except RuntimeError as exc:
                                    rows.append(
                                        {
                                            "case": "reformed_upgrade_mfn_margin",
                                            "mfn_incumbency_T": mfn_incumbency_T,
                                            "sigma_C_T": sigma_C_T,
                                            "fC_T": fC_T,
                                            "upgrade_psi_T": upgrade_psi_T,
                                            "upgrade_psi_comp_T": upgrade_psi_comp_T,
                                            "upgrade_cost_quad_T": upgrade_cost_quad_T,
                                            "upgrade_cost_comp_mult_T": upgrade_cost_comp_mult_T,
                                            "error": str(exc),
                                            "loss": float("inf"),
                                        }
                                    )

    valid = [r for r in rows if "error" not in r and math.isfinite(float(r["loss"]))]
    best = min(valid, key=lambda r: r["loss"]) if valid else None
    t_comp = targets["T"]["conditional_on_ever_us"]["rate"]
    closest_comp = min(valid, key=lambda r: abs(r["T_comp_US"] - t_comp)) if valid else None

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
            "upgrade_requires_US": False,
            "no_qiz_upgrading_allowed": True,
            "mfn_incumbency_margin": "eps lowers MFN US fixed cost and raises QIZ compliance cost",
            "grids": {
                "n_phi": base["n_phi"],
                "n_eps": base["n_eps"],
                "upgrade_intensity_grid_size": base["upgrade_intensity_grid_size"],
                **grid,
            },
        },
        "grid_rows": rows,
        "best": best,
        "closest_compliance": closest_comp,
    }


def write_grid_csv(rows: Iterable[Dict[str, Any]]) -> None:
    cols = [
        "case",
        "mfn_incumbency_T",
        "sigma_C_T",
        "fC_T",
        "upgrade_psi_T",
        "upgrade_psi_comp_T",
        "upgrade_cost_quad_T",
        "upgrade_cost_comp_mult_T",
        "loss",
        "T_comp_US",
        "T_comp_active",
        "T_MFN_US_active",
        "T_US_active",
        "T_RW_active",
        "O_comp_US",
        "T_upgrade_intensity_off_active",
        "T_upgrade_intensity_on_active",
        "T_upgrade_intensity_change",
        "comp_avg_upgrade_intensity",
        "noncomp_avg_upgrade_intensity",
        "upgrade_intensity_gap_comp_noncomp",
        "comp_rw_pct",
        "T_sector_RW_pct",
        "comp_us_pct",
        "welfare_pct",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(summary: Dict[str, Any]) -> None:
    best = summary["best"]
    closest = summary["closest_compliance"]
    targets = summary["targets"]

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Reformed QIZ Calibration\n\n")
        f.write("## Specification\n\n")
        f.write("- One Egypt region: `Q`\n")
        f.write("- QIZ fixed cost is stacked on top of normal US exporting\n")
        f.write("- Upgrading is allowed with and without QIZ\n")
        f.write("- QIZ compliance can add an upgrade complementarity\n")
        f.write("- A latent MFN-incumbency type lowers MFN US fixed costs and raises QIZ compliance costs\n\n")
        f.write("## Main Targets\n\n")
        f.write(
            f"- Textile compliance among ever-US exporters: "
            f"`{targets['T']['conditional_on_ever_us']['rate']:.4f}`\n"
        )
        f.write(
            f"- Textile non-US export growth around first Israeli import: "
            f"`{targets['T_nonus_growth_among_israel_importers']['aggregate_pct']:.1f}%`\n\n"
        )
        if best is None:
            f.write("No stable grid point found.\n")
            return
        f.write("## Best Grid Point\n\n")
        f.write(f"- `mfn_incumbency_T = {best['mfn_incumbency_T']}`\n")
        f.write(f"- `sigma_C_T = {best['sigma_C_T']}`\n")
        f.write(f"- `fC_T = {best['fC_T']}`\n")
        f.write(f"- `upgrade_psi_T = {best['upgrade_psi_T']}`\n")
        f.write(f"- `upgrade_psi_comp_T = {best['upgrade_psi_comp_T']}`\n")
        f.write(f"- `upgrade_cost_quad_T = {best['upgrade_cost_quad_T']}`\n")
        f.write(f"- `upgrade_cost_comp_mult_T = {best['upgrade_cost_comp_mult_T']}`\n")
        f.write(f"- Textile compliance among US exporters: `{best['T_comp_US']:.4f}`\n")
        f.write(f"- Textile compliance among active firms: `{best['T_comp_active']:.4f}`\n")
        f.write(f"- Textile MFN US exporters among active firms: `{best['T_MFN_US_active']:.4f}`\n")
        f.write(f"- Textile US exporters among active firms: `{best['T_US_active']:.4f}`\n")
        f.write(
            f"- Upgrade intensity, no-QIZ to QIZ: "
            f"`{best['T_upgrade_intensity_off_active']:.3f}` to "
            f"`{best['T_upgrade_intensity_on_active']:.3f}`\n"
        )
        f.write(f"- Complier average upgrade intensity: `{best['comp_avg_upgrade_intensity']:.3f}`\n")
        f.write(f"- Noncomplier average upgrade intensity: `{best['noncomp_avg_upgrade_intensity']:.3f}`\n")
        comp_rw = "N/A" if best["comp_rw_pct"] is None else f"{best['comp_rw_pct']:.1f}%"
        f.write(f"- Complier non-US export change: `{comp_rw}`\n")
        f.write(f"- Textile-sector non-US export change: `{best['T_sector_RW_pct']:.1f}%`\n")
        f.write(f"- Welfare change: `{best['welfare_pct']:.3f}%`\n\n")
        if closest is not None and closest is not best:
            f.write("## Closest Compliance Point\n\n")
            f.write(f"- Textile compliance among US exporters: `{closest['T_comp_US']:.4f}`\n")
            f.write(f"- Loss: `{closest['loss']:.3f}`\n\n")
        f.write("## Interpretation\n\n")
        f.write(
            "The reformed specification breaks the previous corner result. Some textile "
            "US exporters now remain MFN because their incumbent US relationship makes "
            "MFN exporting profitable while their QIZ compliance cost is high. Upgrading "
            "is no longer a QIZ-only technology; QIZ raises upgrading by expanding market "
            "access and adding a compliance complementarity.\n"
        )


def main() -> None:
    targets = compute_empirical_targets()
    summary = run_grid(targets)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
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
            f"mfn={best['mfn_incumbency_T']}",
            f"sigma_C={best['sigma_C_T']}",
            f"fC_T={best['fC_T']}",
            f"psi={best['upgrade_psi_T']}",
            f"psi_comp={best['upgrade_psi_comp_T']}",
            f"qcost={best['upgrade_cost_quad_T']}",
            f"qcost_comp_mult={best['upgrade_cost_comp_mult_T']}",
            f"T_comp_US={best['T_comp_US']:.4f}",
            f"T_MFN_US_active={best['T_MFN_US_active']:.4f}",
            f"upgrade_off={best['T_upgrade_intensity_off_active']:.3f}",
            f"upgrade_on={best['T_upgrade_intensity_on_active']:.3f}",
            f"comp_rw_pct={comp_rw_print}",
            f"welfare={best['welfare_pct']:.3f}",
        )


if __name__ == "__main__":
    main()
