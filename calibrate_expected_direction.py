#!/usr/bin/env python3
"""
Search for a coherent benchmark for the project's intended QIZ mechanism:

QIZ use -> larger US payoff -> upgrading -> higher non-US exports for the
same treated textile firms -> welfare comparison in general equilibrium.

This script keeps the estimated/data-backed primitives fixed and searches over
the textile-side mechanism margins that remain weakly identified:
- textile domestic fixed cost
- textile entry costs in Q and N
- textile compliance cost mean / heterogeneity
- compliance-driven upgrade payoff / curvature

Outputs:
- expected_direction_calibration_summary.json
- expected_direction_calibrated_params.json
- expected_direction_qiz_vs_noqiz.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List

import numpy as np

from calibrate_fixed_costs import TARGETS, get_participation, get_qiz_firm_share
import qiz_model_ge as m


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "expected_direction_calibration_summary.json")
OUT_PARAMS = os.path.join(ROOT, "expected_direction_calibrated_params.json")
OUT_CSV = os.path.join(ROOT, "expected_direction_qiz_vs_noqiz.csv")

TEXTILE_QIZ_SHARE_MIN = 0.5 * TARGETS["qiz_firm_share"]["T"]

GRID = {
    "f_dom_T": [0.80, 0.90, 1.00],
    "f_entry_Q_T": [0.30, 0.50, 0.70],
    "f_entry_N_T": [1.00, 1.50, 2.00],
    "fC_mean_T": [6.00, 8.00, 10.00],
    "sigma_C_T": [0.35, 0.50, 0.65],
    "upgrade_psi_comp_T": [0.06, 0.08],
    "upgrade_cost_quad_T": [6.00, 8.00, 10.00],
}

SEEDS = [
    {
        "f_dom_T": 0.90,
        "f_entry_Q_T": 0.50,
        "f_entry_N_T": 2.00,
        "fC_mean_T": 2.00,
        "sigma_C_T": 0.25,
        "upgrade_psi_comp_T": 0.08,
        "upgrade_cost_quad_T": 4.50,
    },
    {
        "f_dom_T": 0.90,
        "f_entry_Q_T": 0.30,
        "f_entry_N_T": 2.00,
        "fC_mean_T": 8.00,
        "sigma_C_T": 0.65,
        "upgrade_psi_comp_T": 0.06,
        "upgrade_cost_quad_T": 8.00,
    },
    {
        "f_dom_T": 0.90,
        "f_entry_Q_T": 0.50,
        "f_entry_N_T": 1.50,
        "fC_mean_T": 6.00,
        "sigma_C_T": 0.35,
        "upgrade_psi_comp_T": 0.06,
        "upgrade_cost_quad_T": 6.00,
    },
]


def exact_penalty(actual: float, target: float, scale: float, weight: float) -> float:
    if not np.isfinite(actual):
        return 1.0e9
    return weight * (((actual - target) / scale) ** 2)


def floor_penalty(actual: float, floor: float, scale: float, weight: float) -> float:
    if not np.isfinite(actual):
        return 1.0e9
    shortfall = max(0.0, floor - actual)
    return weight * ((shortfall / scale) ** 2)


def ceiling_penalty(actual: float, ceiling: float, scale: float, weight: float) -> float:
    if not np.isfinite(actual):
        return 1.0e9
    excess = max(0.0, actual - ceiling)
    return weight * ((excess / scale) ** 2)


def build_base_params(calibration_mode: bool) -> Dict[str, Any]:
    p = m.build_params_from_preset("expected_direction_base")

    if calibration_mode:
        # Keep enough resolution so heterogeneity remains informative while
        # still making a local coordinate search affordable.
        p["n_phi"] = 15
        p["n_eps"] = 5
        p["goods_max_iter"] = 120
        p["outer_max_iter"] = 250
        p["outer_tol"] = max(float(p["outer_tol"]), 2.0e-3)
        p["outer_cycle_tol"] = max(float(p.get("outer_cycle_tol", 2.0e-3)), 8.0e-3)

    # Keep non-textiles pinned to the original stage-2 entry normalization.
    p["f_entry"][("Q", "O")] = 0.30
    p["f_entry"][("N", "O")] = 1.00
    return p


def apply_state(p: Dict[str, Any], state: Dict[str, float]) -> None:
    for r in p["regions"]:
        p["f_dom"][(r, "T")] = state["f_dom_T"]

    p["f_entry"][("Q", "T")] = state["f_entry_Q_T"]
    p["f_entry"][("N", "T")] = state["f_entry_N_T"]
    p["fC_mean"]["T"] = state["fC_mean_T"]
    p["sigma_C"]["T"] = state["sigma_C_T"]
    p["upgrade_psi_comp"]["T"] = state["upgrade_psi_comp_T"]
    p["upgrade_cost_quad"]["T"] = state["upgrade_cost_quad_T"]


def compare_with_warm(
    p: Dict[str, Any], warm_on: Dict[str, Any] | None = None, warm_off: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    try:
        on = m.solve_equilibrium(p, qiz_on=True, disable_upgrade=False, initial_state=warm_on, verbose=False)
    except RuntimeError:
        on = m.solve_equilibrium(p, qiz_on=True, disable_upgrade=False, verbose=False)

    try:
        off = m.solve_equilibrium(p, qiz_on=False, disable_upgrade=False, initial_state=warm_off or on, verbose=False)
    except RuntimeError:
        off = m.solve_equilibrium(p, qiz_on=False, disable_upgrade=False, initial_state=on, verbose=False)

    tr_on = m.summarize_trade(on, p)
    tr_off = m.summarize_trade(off, p)

    dec: Dict[tuple[str, str], Dict[str, float]] = {}
    for key in tr_on["exports"]:
        X1 = tr_on["exports"][key]
        X0 = tr_off["exports"][key]
        N1 = tr_on["exporter_masses"][key]
        N0 = tr_off["exporter_masses"][key]
        x1 = tr_on["avg_revenue_per_exporter"][key]
        x0 = tr_off["avg_revenue_per_exporter"][key]

        dX = X1 - X0
        ext = (N1 - N0) * x0
        inten = N0 * (x1 - x0)
        interact = (N1 - N0) * (x1 - x0)
        pct = 100.0 * dX / max(abs(X0), 1.0e-12)

        dec[key] = {
            "on": X1,
            "off": X0,
            "delta": dX,
            "pct_change": pct,
            "extensive_component": ext,
            "intensive_component": inten,
            "interaction_component": interact,
        }

    return {"on": on, "off": off, "trade_on": tr_on, "trade_off": tr_off, "decomposition": dec}


def extract_metrics(comp: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, float]:
    on = comp["on"]
    off = comp["off"]
    moms = m.collect_stylized_moments(comp, p)
    part_t = get_participation(off, "T")
    part_o = get_participation(off, "O")
    qt_on = on["moments"][("Q", "T")]
    qt_off = off["moments"][("Q", "T")]

    return {
        "dom_only_T": float(part_t["dom_only"]),
        "dom_only_O": float(part_o["dom_only"]),
        "exp_US_T": float(part_t["exp_US"]),
        "exp_US_O": float(part_o["exp_US"]),
        "exp_RW_T": float(part_t["exp_RW"]),
        "exp_RW_O": float(part_o["exp_RW"]),
        "qiz_firm_share_T": float(get_qiz_firm_share(off, "T")),
        "qiz_firm_share_O": float(get_qiz_firm_share(off, "O")),
        "uptake_T": float(qt_on["compliance_share_among_US_exporters"]),
        "upgrade_T": float(qt_on["upgrade_share_among_active"]),
        "qt_active_on": float(qt_on["active_share"]),
        "qt_us_share_on": float(qt_on["US_export_share_among_active"]),
        "qt_active_off": float(qt_off["active_share"]),
        "qt_us_share_off": float(qt_off["US_export_share_among_active"]),
        "nt_mass_off": float(off["M"][("N", "T")]),
        "tier1_comp_us_logchg": float(moms["tier1_comp_us_logchg"]),
        "tier1_comp_rw_logchg": float(moms["tier1_comp_rw_logchg"]),
        "tier2_emp_qt_logchg": float(moms["tier2_emp_qt_logchg"]),
        "tier2_prod_qt_logchg": float(moms["tier2_prod_qt_logchg"]),
        "tier3_qt_us_logchg": float(moms["tier3_qt_us_logchg"]),
        "tier3_total_us_logchg": float(moms["tier3_total_us_logchg"]),
        "tier3_welfare_logchg": float(moms["tier3_welfare_logchg"]),
        "micro_comp_us_pct": float(moms["micro_comp_us_pct"]),
        "micro_comp_rw_pct": float(moms["micro_comp_rw_pct"]),
        "agg_total_us_pct": float(moms["agg_total_us_pct"]),
        "agg_q_t_us_pct": float(moms["agg_q_t_us_pct"]),
        "real_wage_q_pct": float(moms["target_real_wage_q_pct"]),
        "welfare_pct": float(100.0 * (on["welfare"] / off["welfare"] - 1.0)),
    }


def loss_breakdown(metrics: Dict[str, float]) -> Dict[str, float]:
    target_us = math.log(1.5)
    target_rw = math.log(1.1)
    return {
        # Pre-QIZ participation discipline.
        "dom_only_T": exact_penalty(metrics["dom_only_T"], TARGETS["dom_only"]["T"], scale=0.03, weight=1.1),
        "dom_only_O": exact_penalty(metrics["dom_only_O"], TARGETS["dom_only"]["O"], scale=0.02, weight=1.0),
        "exp_US_T": exact_penalty(metrics["exp_US_T"], TARGETS["exp_US"]["T"], scale=0.02, weight=1.2),
        "exp_US_O": exact_penalty(metrics["exp_US_O"], TARGETS["exp_US"]["O"], scale=0.02, weight=1.0),
        "exp_RW_T": exact_penalty(metrics["exp_RW_T"], TARGETS["exp_RW"]["T"], scale=0.03, weight=1.0),
        "exp_RW_O": exact_penalty(metrics["exp_RW_O"], TARGETS["exp_RW"]["O"], scale=0.03, weight=1.0),
        # Region concentration.
        "qiz_firm_share_T_soft": exact_penalty(
            metrics["qiz_firm_share_T"], TARGETS["qiz_firm_share"]["T"], scale=0.10, weight=1.00
        ),
        "qiz_firm_share_T_floor": floor_penalty(
            metrics["qiz_firm_share_T"], TEXTILE_QIZ_SHARE_MIN, scale=0.05, weight=12.0
        ),
        "qiz_firm_share_T_ceiling": ceiling_penalty(metrics["qiz_firm_share_T"], 0.98, scale=0.03, weight=4.0),
        "qiz_firm_share_O": exact_penalty(
            metrics["qiz_firm_share_O"], TARGETS["qiz_firm_share"]["O"], scale=0.06, weight=1.0
        ),
        # QIZ-on uptake / upgrading moments.
        "uptake_T": exact_penalty(metrics["uptake_T"], TARGETS["compliance_rate"], scale=0.06, weight=1.8),
        "upgrade_T": exact_penalty(metrics["upgrade_T"], TARGETS["upgrading_rate"], scale=0.08, weight=1.5),
        # Same-type firm mechanism.
        "comp_us_direction": floor_penalty(metrics["tier1_comp_us_logchg"], target_us, scale=0.25, weight=3.0),
        "comp_rw_direction": floor_penalty(metrics["tier1_comp_rw_logchg"], target_rw, scale=0.20, weight=3.0),
        # Treated textiles should not contract sharply while fitting micro moments.
        "qt_emp_nonnegative": floor_penalty(metrics["tier2_emp_qt_logchg"], 0.0, scale=0.25, weight=2.0),
        "qt_prod_nonnegative": floor_penalty(metrics["tier2_prod_qt_logchg"], 0.0, scale=0.25, weight=2.0),
        # Aggregate discipline.
        "qt_us_nonnegative": floor_penalty(metrics["tier3_qt_us_logchg"], 0.0, scale=0.15, weight=1.8),
        "total_us_nonnegative": floor_penalty(metrics["tier3_total_us_logchg"], 0.0, scale=0.12, weight=1.2),
        "welfare_nonnegative": floor_penalty(metrics["tier3_welfare_logchg"], 0.0, scale=0.05, weight=1.0),
        # Avoid pathological corners.
        "uptake_floor": floor_penalty(metrics["uptake_T"], 0.05, scale=0.03, weight=1.0),
        "uptake_ceiling": ceiling_penalty(metrics["uptake_T"], 0.95, scale=0.03, weight=1.0),
        "upgrade_floor": floor_penalty(metrics["upgrade_T"], 0.05, scale=0.03, weight=0.8),
        "upgrade_ceiling": ceiling_penalty(metrics["upgrade_T"], 0.95, scale=0.03, weight=0.8),
        "qt_active_floor": floor_penalty(metrics["qt_active_on"], 0.05, scale=0.03, weight=0.8),
        "qt_us_share_floor": floor_penalty(metrics["qt_us_share_on"], 0.05, scale=0.03, weight=0.6),
        "qt_active_off_floor": floor_penalty(metrics["qt_active_off"], 0.05, scale=0.03, weight=2.0),
        "qt_us_share_off_floor": floor_penalty(metrics["qt_us_share_off"], 0.05, scale=0.03, weight=6.0),
        "nt_mass_floor": floor_penalty(metrics["nt_mass_off"], 1.0, scale=10.0, weight=0.8),
    }


def evaluate_state(
    state: Dict[str, float],
    calibration_mode: bool,
    warm_on: Dict[str, Any] | None = None,
    warm_off: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    p = build_base_params(calibration_mode=calibration_mode)
    apply_state(p, state)

    try:
        comp = compare_with_warm(p, warm_on=warm_on, warm_off=warm_off)
    except RuntimeError as exc:
        return {"ok": False, "state": state, "error": str(exc), "loss": float("inf")}

    metrics = extract_metrics(comp, p)
    breakdown = loss_breakdown(metrics)
    return {
        "ok": True,
        "state": state,
        "p": p,
        "comp": comp,
        "metrics": metrics,
        "loss_breakdown": breakdown,
        "loss": float(sum(breakdown.values())),
    }


def summarize_exports(comp: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, float]:
    tr_on = m.summarize_trade(comp["on"], p)
    tr_off = m.summarize_trade(comp["off"], p)
    return {
        "Q_US_on": float(tr_on["exports"][("Q", "US")]),
        "Q_US_off": float(tr_off["exports"][("Q", "US")]),
        "Q_RW_on": float(tr_on["exports"][("Q", "RW")]),
        "Q_RW_off": float(tr_off["exports"][("Q", "RW")]),
        "N_US_on": float(tr_on["exports"][("N", "US")]),
        "N_US_off": float(tr_off["exports"][("N", "US")]),
        "N_RW_on": float(tr_on["exports"][("N", "RW")]),
        "N_RW_off": float(tr_off["exports"][("N", "RW")]),
        "total_US_on": float(sum(tr_on["exports"][(r, "US")] for r in p["regions"])),
        "total_US_off": float(sum(tr_off["exports"][(r, "US")] for r in p["regions"])),
        "total_RW_on": float(sum(tr_on["exports"][(r, "RW")] for r in p["regions"])),
        "total_RW_off": float(sum(tr_off["exports"][(r, "RW")] for r in p["regions"])),
    }


def better(candidate: Dict[str, Any], incumbent: Dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    return candidate["loss"] < incumbent["loss"] - 1.0e-12


def print_update(prefix: str, rec: Dict[str, Any]) -> None:
    print(
        f"{prefix}: loss={rec['loss']:.4f} state={rec['state']} "
        f"metrics={{uptake={rec['metrics']['uptake_T']:.3f}, "
        f"upgrade={rec['metrics']['upgrade_T']:.3f}, "
        f"qshareT={rec['metrics']['qiz_firm_share_T']:.3f}, "
        f"NTmass={rec['metrics']['nt_mass_off']:.2e}, "
        f"US={rec['metrics']['micro_comp_us_pct']:.1f}, "
        f"RW={rec['metrics']['micro_comp_rw_pct']:.1f}, "
        f"aggUS={rec['metrics']['agg_total_us_pct']:.1f}, "
        f"welfare={rec['metrics']['welfare_pct']:.3f}}}"
    )


def coordinate_descent(seed: Dict[str, float], rounds: int = 2) -> Dict[str, Any]:
    current = {k: float(v) for k, v in seed.items()}
    best = evaluate_state(current, calibration_mode=True)
    if not best.get("ok", False):
        raise RuntimeError(f"Seed did not converge: {seed}")

    print_update("seed", best)

    for round_idx in range(rounds):
        improved = False
        warm_on = best["comp"]["on"]
        warm_off = best["comp"]["off"]
        for key, values in GRID.items():
            local_best = best
            for value in values:
                cand_state = dict(current)
                cand_state[key] = float(value)
                rec = evaluate_state(cand_state, calibration_mode=True, warm_on=warm_on, warm_off=warm_off)
                if rec.get("ok", False) and better(rec, local_best):
                    local_best = rec
            if better(local_best, best):
                best = local_best
                current = dict(best["state"])
                improved = True
                warm_on = best["comp"]["on"]
                warm_off = best["comp"]["off"]
                print_update(f"round {round_idx + 1} {key}", best)
        if not improved:
            break
    return best


def directional_solution_flags(metrics: Dict[str, float]) -> Dict[str, bool]:
    return {
        "micro_us_positive": bool(metrics["micro_comp_us_pct"] > 0.0),
        "micro_rw_positive": bool(metrics["micro_comp_rw_pct"] > 0.0),
        "agg_total_us_nonnegative": bool(metrics["agg_total_us_pct"] >= 0.0),
        "agg_q_t_us_nonnegative": bool(metrics["agg_q_t_us_pct"] >= 0.0),
        "welfare_nonnegative": bool(metrics["welfare_pct"] >= 0.0),
        "uptake_interior": bool(0.05 < metrics["uptake_T"] < 0.95),
        "upgrade_interior": bool(0.05 < metrics["upgrade_T"] < 0.95),
        "nt_mass_positive": bool(metrics["nt_mass_off"] > 1.0e-6),
    }


def refine_high_resolution(best: Dict[str, Any]) -> Dict[str, Any]:
    state = best["state"]
    p = build_base_params(calibration_mode=False)
    apply_state(p, state)

    comp = compare_with_warm(p, warm_on=best["comp"]["on"], warm_off=best["comp"]["off"])
    metrics = extract_metrics(comp, p)
    breakdown = loss_breakdown(metrics)

    cf_prod = m.counterfactual_shutdown_productivity(p, baseline=comp["on"])
    welfare_baseline = float(cf_prod["baseline"]["welfare"])
    welfare_no_prod = float(cf_prod["no_productivity"]["welfare"])
    metrics["welfare_gain_from_productivity_channel_pct"] = float(
        100.0 * (welfare_baseline / max(welfare_no_prod, 1.0e-12) - 1.0)
    )

    return {
        "state": state,
        "p": p,
        "comp": comp,
        "metrics": metrics,
        "loss_breakdown": breakdown,
        "loss": float(sum(breakdown.values())),
        "productivity_cf": {
            "welfare_baseline": welfare_baseline,
            "welfare_no_productivity": welfare_no_prod,
            "welfare_gain_pct": metrics["welfare_gain_from_productivity_channel_pct"],
            "qt_baseline": cf_prod["baseline"]["moments"][("Q", "T")],
            "qt_no_productivity": cf_prod["no_productivity"]["moments"][("Q", "T")],
        },
    }


def write_csv(best: Dict[str, Any]) -> None:
    comp = best["comp"]
    p = best["p"]
    on = comp["on"]
    off = comp["off"]
    exports = summarize_exports(comp, p)

    rows = [
        ["metric", "qiz_on", "qiz_off"],
        ["welfare", on["welfare"], off["welfare"]],
        ["real_wage_Q", on["w"]["Q"] / on["goods"]["P_EG"], off["w"]["Q"] / off["goods"]["P_EG"]],
        ["real_wage_N", on["w"]["N"] / on["goods"]["P_EG"], off["w"]["N"] / off["goods"]["P_EG"]],
        ["Q_employment_share", on["Ls"]["Q"] / p["L_total"], off["Ls"]["Q"] / p["L_total"]],
        ["QT_active_share", on["moments"][("Q", "T")]["active_share"], off["moments"][("Q", "T")]["active_share"]],
        ["QT_compliance_share_US_exporters", on["moments"][("Q", "T")]["compliance_share_among_US_exporters"], 0.0],
        ["QT_upgrade_share_active", on["moments"][("Q", "T")]["upgrade_share_among_active"], off["moments"][("Q", "T")]["upgrade_share_among_active"]],
        ["Q_US_exports", exports["Q_US_on"], exports["Q_US_off"]],
        ["Q_RW_exports", exports["Q_RW_on"], exports["Q_RW_off"]],
        ["N_US_exports", exports["N_US_on"], exports["N_US_off"]],
        ["total_US_exports", exports["total_US_on"], exports["total_US_off"]],
        ["total_RW_exports", exports["total_RW_on"], exports["total_RW_off"]],
        ["micro_comp_us_pct", best["metrics"]["micro_comp_us_pct"], ""],
        ["micro_comp_rw_pct", best["metrics"]["micro_comp_rw_pct"], ""],
        ["welfare_gain_from_productivity_channel_pct", best["metrics"]["welfare_gain_from_productivity_channel_pct"], ""],
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    best: Dict[str, Any] | None = None

    for idx, seed in enumerate(SEEDS, start=1):
        print(f"Starting seed {idx}/{len(SEEDS)}: {seed}")
        try:
            rec = coordinate_descent(seed, rounds=2)
        except RuntimeError as exc:
            print(f"Skipping seed {idx}: {exc}")
            continue
        if better(rec, best):
            best = rec

    if best is None:
        raise RuntimeError("No converged candidate found.")

    try:
        final = refine_high_resolution(best)
        final_mode = "high_resolution"
    except RuntimeError as exc:
        print(f"High-resolution refine failed, keeping calibration candidate: {exc}")
        final = best
        final_mode = "calibration_grid_only"
        final["metrics"]["welfare_gain_from_productivity_channel_pct"] = float("nan")
        final["productivity_cf"] = None

    final_exports = summarize_exports(final["comp"], final["p"])
    flags = directional_solution_flags(final["metrics"])

    summary = {
        "search_grid": GRID,
        "best_state_calibration_grid": best["state"],
        "best_loss_calibration_grid": best["loss"],
        "best_state_final": final["state"],
        "final_mode": final_mode,
        "final_loss": final["loss"],
        "final_metrics": final["metrics"],
        "final_loss_breakdown": final["loss_breakdown"],
        "final_exports": final_exports,
        "final_productivity_counterfactual": final.get("productivity_cf"),
        "directional_solution_flags": flags,
        "found_full_directional_solution": bool(all(flags.values())),
        "final_qt_on": final["comp"]["on"]["moments"][("Q", "T")],
        "final_qt_off": final["comp"]["off"]["moments"][("Q", "T")],
        "final_M_on": {str(k): v for k, v in final["comp"]["on"]["M"].items()},
        "final_M_off": {str(k): v for k, v in final["comp"]["off"]["M"].items()},
        "final_qiz_vs_noqiz": {
            "welfare_on": final["comp"]["on"]["welfare"],
            "welfare_off": final["comp"]["off"]["welfare"],
            "welfare_pct": final["metrics"]["welfare_pct"],
            "wQ_on": final["comp"]["on"]["w"]["Q"],
            "wQ_off": final["comp"]["off"]["w"]["Q"],
            "wN_on": final["comp"]["on"]["w"]["N"],
            "wN_off": final["comp"]["off"]["w"]["N"],
        },
    }

    params_out = {
        "preset": "expected_direction_base",
        "state": final["state"],
        "mechanism_switches": {
            "roo_cost_formula": final["p"]["roo_cost_formula"],
            "roo_cost_scope": final["p"]["roo_cost_scope"],
            "use_admin_wedge": final["p"]["use_admin_wedge"],
            "upgrade_mode": final["p"]["upgrade_mode"],
            "upgrade_requires_US": final["p"]["upgrade_requires_US"],
        },
        "calibrated_values": {
            "f_dom_T": final["p"]["f_dom"][("Q", "T")],
            "f_dom_O": final["p"]["f_dom"][("Q", "O")],
            "f_export_US_T": final["p"]["f_export"][("Q", "US", "T")],
            "f_export_RW_T": final["p"]["f_export"][("Q", "RW", "T")],
            "f_export_US_O": final["p"]["f_export"][("Q", "US", "O")],
            "f_export_RW_O": final["p"]["f_export"][("Q", "RW", "O")],
            "f_entry_Q_T": final["p"]["f_entry"][("Q", "T")],
            "f_entry_N_T": final["p"]["f_entry"][("N", "T")],
            "f_entry_Q_O": final["p"]["f_entry"][("Q", "O")],
            "f_entry_N_O": final["p"]["f_entry"][("N", "O")],
            "fC_mean_T": final["p"]["fC_mean"]["T"],
            "sigma_C_T": final["p"]["sigma_C"]["T"],
            "upgrade_psi_T": final["p"]["upgrade_psi"]["T"],
            "upgrade_psi_comp_T": final["p"]["upgrade_psi_comp"]["T"],
            "upgrade_cost_quad_T": final["p"]["upgrade_cost_quad"]["T"],
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(OUT_PARAMS, "w", encoding="utf-8") as f:
        json.dump(params_out, f, indent=2)
    write_csv(final)

    print(f"Saved summary: {OUT_JSON}")
    print(f"Saved params: {OUT_PARAMS}")
    print(f"Saved comparison CSV: {OUT_CSV}")
    print(f"Found full directional solution: {all(flags.values())}")


if __name__ == "__main__":
    main()
