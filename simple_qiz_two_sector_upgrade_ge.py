#!/usr/bin/env python3
"""
Simplified QIZ GE experiment with two sectors and upgrading:

- one region: Q
- two sectors: T and O
- endogenous entry and export participation
- QIZ treatment active only for textiles
- upgrading active only for textiles
- no interregional labor mobility

This is a diagnostic bridge between the one-sector stripped model and the full
two-region model.
"""

from __future__ import annotations

import csv
import json
import os
from copy import deepcopy
from typing import Any, Dict

import qiz_model_ge as m


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "simple_qiz_two_sector_upgrade_summary.json")
OUT_CSV = os.path.join(ROOT, "simple_qiz_two_sector_upgrade_qiz_vs_noqiz.csv")

TARGETS = {
    "dom_only": {"T": 0.8927, "O": 0.8492},
    "exp_US": {"T": 0.0575, "O": 0.0112},
    "exp_RW": {"T": 0.0690, "O": 0.1425},
    "compliance_rate_T": 0.321,
    "upgrading_rate_T": 0.491,
}

GRID_STAGE1 = {
    "f_dom_T": [0.8, 1.0, 1.2, 1.5],
    "f_export_US_T": [8.0, 12.0, 18.0, 25.0, 35.0],
    "f_export_RW_T": [12.0, 20.0, 35.0, 50.0, 80.0],
    "f_dom_O": [0.6, 0.8, 1.0, 1.2],
    "f_export_US_O": [2.0, 3.5, 5.0, 7.5, 10.0],
    "f_export_RW_O": [2.0, 3.5, 5.0, 7.5, 10.0],
}

GRID_STAGE23 = {
    "fC_mean_T": [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0],
    "f_upgrade_T": [0.2, 0.4, 0.6, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0],
}


def build_simple_params() -> Dict[str, Any]:
    p = deepcopy(m.params_defensible())

    p["regions"] = ["Q"]
    p["sectors"] = ["T", "O"]
    p["dests"] = ["EG", "US", "RW"]

    # One region => fixed labor supply, no regional reallocation.
    p["L_total"] = 1.0
    p["L_Q_share"] = 1.0
    p["kappa"] = 1.0

    # Cleaner ROO implementation.
    p["roo_cost_formula"] = "normalized"
    p["roo_cost_scope"] = "US_only"
    p["use_admin_wedge"] = False
    p["xi_admin"]["T"] = 0.0
    p["xi_admin"]["O"] = 0.0

    # Policy active only for textiles in this diagnostic model.
    p["t_mfn"]["O"] = 0.0
    p["gamma"]["O"] = 0.0
    p["p_il"]["O"] = 1.0
    p["fC_mean"]["O"] = 1.0e6
    p["sigma_C"]["O"] = 0.0

    # Upgrading active only for textiles and tied to US service.
    p["upgrade_mode"] = "binary"
    p["upgrade_requires_US"] = True
    p["delta"]["T"] = 1.178
    p["delta"]["O"] = 1.0
    p["f_upgrade"]["T"] = 1.8
    p["f_upgrade"]["O"] = 1.0e6

    # Normalize entry costs in the stripped-down model.
    p["f_entry"][("Q", "T")] = 1.0
    p["f_entry"][("Q", "O")] = 1.0

    # Participation/compliance starting values; calibrated below.
    p["f_dom"][("Q", "T")] = 1.0
    p["f_export"][("Q", "US", "T")] = 12.0
    p["f_export"][("Q", "RW", "T")] = 20.0
    p["f_dom"][("Q", "O")] = 0.8
    p["f_export"][("Q", "US", "O")] = 3.5
    p["f_export"][("Q", "RW", "O")] = 3.5
    p["fC_mean"]["T"] = 8.0
    p["sigma_C"]["T"] = 0.35

    # Keep only the one-region wedges and foreign shifters.
    p["d_iceberg"] = {
        ("Q", "EG", "T"): p["d_iceberg"][("Q", "EG", "T")],
        ("Q", "US", "T"): p["d_iceberg"][("Q", "US", "T")],
        ("Q", "RW", "T"): p["d_iceberg"][("Q", "RW", "T")],
        ("Q", "EG", "O"): p["d_iceberg"][("Q", "EG", "O")],
        ("Q", "US", "O"): p["d_iceberg"][("Q", "US", "O")],
        ("Q", "RW", "O"): p["d_iceberg"][("Q", "RW", "O")],
    }
    p["e_ratio_foreign"] = {
        ("US", "T"): p["e_ratio_foreign"][("US", "T")],
        ("RW", "T"): p["e_ratio_foreign"][("RW", "T")],
        ("US", "O"): p["e_ratio_foreign"][("US", "O")],
        ("RW", "O"): p["e_ratio_foreign"][("RW", "O")],
    }
    p["E_foreign"] = {k: 1.0 for k in p["e_ratio_foreign"]}
    p["P_foreign"] = {("US", "T"): 1.0, ("RW", "T"): 1.0, ("US", "O"): 1.0, ("RW", "O"): 1.0}

    # Moderate resolution; still cheap.
    p["n_phi"] = 100
    p["n_eps"] = 7
    p["goods_max_iter"] = 220
    p["outer_max_iter"] = 420
    p["outer_tol"] = 1.0e-4
    p["outer_cycle_tol"] = 5.0e-3
    return p


def participation_from_solution(sol: Dict[str, Any], sector: str) -> Dict[str, float]:
    mom = sol["moments"][("Q", sector)]
    return {
        "dom_only": float(mom["domestic_only_share_among_active"]),
        "exp_US": float(mom["US_export_share_among_active"]),
        "exp_RW": float(mom["RW_export_share_among_active"]),
        "active_share": float(mom["active_share"]),
    }


def stage1_loss(part_t: Dict[str, float], part_o: Dict[str, float]) -> float:
    return (
        ((part_t["dom_only"] - TARGETS["dom_only"]["T"]) / 0.03) ** 2
        + ((part_t["exp_US"] - TARGETS["exp_US"]["T"]) / 0.02) ** 2
        + ((part_t["exp_RW"] - TARGETS["exp_RW"]["T"]) / 0.02) ** 2
        + ((part_o["dom_only"] - TARGETS["dom_only"]["O"]) / 0.03) ** 2
        + ((part_o["exp_US"] - TARGETS["exp_US"]["O"]) / 0.02) ** 2
        + ((part_o["exp_RW"] - TARGETS["exp_RW"]["O"]) / 0.03) ** 2
    )


def stage23_loss(uptake_t: float, upgrade_t: float) -> float:
    return (
        ((uptake_t - TARGETS["compliance_rate_T"]) / 0.04) ** 2
        + ((upgrade_t - TARGETS["upgrading_rate_T"]) / 0.06) ** 2
    )


def set_stage1_costs(p: Dict[str, Any], state: Dict[str, float]) -> None:
    p["f_dom"][("Q", "T")] = state["f_dom_T"]
    p["f_export"][("Q", "US", "T")] = state["f_export_US_T"]
    p["f_export"][("Q", "RW", "T")] = state["f_export_RW_T"]
    p["f_dom"][("Q", "O")] = state["f_dom_O"]
    p["f_export"][("Q", "US", "O")] = state["f_export_US_O"]
    p["f_export"][("Q", "RW", "O")] = state["f_export_RW_O"]


def default_initial_state(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "w": {r: 1.0 for r in p["regions"]},
        "M": {(r, s): 0.01 for r in p["regions"] for s in p["sectors"]},
        "goods": {"P_EG_s": {s: 1.0 for s in p["sectors"]}},
    }


def run_off_equilibrium(base_p: Dict[str, Any], state: Dict[str, float], warm: Dict[str, Any] | None = None) -> Dict[str, Any]:
    p = deepcopy(base_p)
    set_stage1_costs(p, state)
    try:
        sol = m.solve_equilibrium(
            p,
            qiz_on=False,
            disable_upgrade=True,
            initial_state=warm or default_initial_state(p),
            verbose=False,
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc), "loss": float("inf"), "state": dict(state)}
    part_t = participation_from_solution(sol, "T")
    part_o = participation_from_solution(sol, "O")
    return {
        "ok": True,
        "p": p,
        "sol": sol,
        "part_t": part_t,
        "part_o": part_o,
        "loss": stage1_loss(part_t, part_o),
        "state": dict(state),
    }


def coordinate_stage1(base_p: Dict[str, Any]) -> Dict[str, Any]:
    current = {
        "f_dom_T": 1.0,
        "f_export_US_T": 12.0,
        "f_export_RW_T": 20.0,
        "f_dom_O": 0.8,
        "f_export_US_O": 3.5,
        "f_export_RW_O": 3.5,
    }
    best = run_off_equilibrium(base_p, current)
    if not best.get("ok", False):
        raise RuntimeError(f"Initial stage1 state did not converge: {best.get('error')}")

    for round_idx in range(3):
        improved = False
        warm = best["sol"]
        for key, values in GRID_STAGE1.items():
            local_best = best
            for value in values:
                candidate = dict(current)
                candidate[key] = float(value)
                rec = run_off_equilibrium(base_p, candidate, warm=warm)
                if rec.get("ok", False) and rec["loss"] < local_best["loss"] - 1.0e-12:
                    local_best = rec
            if local_best["loss"] < best["loss"] - 1.0e-12:
                best = local_best
                current = dict(best["state"])
                warm = best["sol"]
                improved = True
                print(
                    f"stage1 round {round_idx + 1} {key}: "
                    f"loss={best['loss']:.4f} state={best['state']} "
                    f"T={best['part_t']} O={best['part_o']}"
                )
        if not improved:
            break
    return best


def calibrate_compliance_upgrade(base_p: Dict[str, Any], stage1: Dict[str, Any]) -> Dict[str, Any]:
    best = None
    warm = stage1["sol"]
    for fC in GRID_STAGE23["fC_mean_T"]:
        for fU in GRID_STAGE23["f_upgrade_T"]:
            p = deepcopy(base_p)
            set_stage1_costs(p, stage1["state"])
            p["fC_mean"]["T"] = float(fC)
            p["f_upgrade"]["T"] = float(fU)
            try:
                on = m.solve_equilibrium(p, qiz_on=True, disable_upgrade=False, initial_state=warm, verbose=False)
                off = m.solve_equilibrium(p, qiz_on=False, disable_upgrade=False, initial_state=on, verbose=False)
            except RuntimeError:
                continue
            uptake_t = float(on["moments"][("Q", "T")]["compliance_share_among_US_exporters"])
            upgrade_t = float(on["moments"][("Q", "T")]["upgrade_share_among_active"])
            rec = {
                "p": p,
                "on": on,
                "off": off,
                "fC_mean_T": float(fC),
                "f_upgrade_T": float(fU),
                "uptake_T": uptake_t,
                "upgrade_T": upgrade_t,
                "loss": stage23_loss(uptake_t, upgrade_t),
            }
            if best is None or rec["loss"] < best["loss"] - 1.0e-12:
                best = rec
                print(
                    f"stage23 fC_mean_T={fC:.3f} f_upgrade_T={fU:.3f}: "
                    f"loss={rec['loss']:.4f} uptake={uptake_t:.4f} upgrade={upgrade_t:.4f}"
                )
    if best is None:
        raise RuntimeError("Compliance/upgrade calibration failed.")
    return best


def trade_summary(sol: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, float]:
    tr = m.summarize_trade(sol, p)
    sec = m.summarize_trade_by_sector(sol, p)
    return {
        "Q_US_total": float(tr["exports"][("Q", "US")]),
        "Q_RW_total": float(tr["exports"][("Q", "RW")]),
        "Q_US_T": float(sec[("Q", "T", "US")]),
        "Q_RW_T": float(sec[("Q", "T", "RW")]),
        "Q_US_O": float(sec[("Q", "O", "US")]),
        "Q_RW_O": float(sec[("Q", "O", "RW")]),
        "US_exporters_total": float(tr["exporter_masses"][("Q", "US")]),
        "RW_exporters_total": float(tr["exporter_masses"][("Q", "RW")]),
        "avg_US_rev_per_exporter": float(tr["avg_revenue_per_exporter"][("Q", "US")]),
        "avg_RW_rev_per_exporter": float(tr["avg_revenue_per_exporter"][("Q", "RW")]),
    }


def write_csv(on: Dict[str, Any], off: Dict[str, Any], summary: Dict[str, Any]) -> None:
    rows = [
        ["metric", "qiz_on", "qiz_off"],
        ["welfare", summary["welfare_on"], summary["welfare_off"]],
        ["price_index", on["goods"]["P_EG"], off["goods"]["P_EG"]],
        ["entry_mass_QT", on["M"][("Q", "T")], off["M"][("Q", "T")]],
        ["entry_mass_QO", on["M"][("Q", "O")], off["M"][("Q", "O")]],
        ["active_share_QT", on["moments"][("Q", "T")]["active_share"], off["moments"][("Q", "T")]["active_share"]],
        ["active_share_QO", on["moments"][("Q", "O")]["active_share"], off["moments"][("Q", "O")]["active_share"]],
        ["US_export_share_QT", on["moments"][("Q", "T")]["US_export_share_among_active"], off["moments"][("Q", "T")]["US_export_share_among_active"]],
        ["RW_export_share_QT", on["moments"][("Q", "T")]["RW_export_share_among_active"], off["moments"][("Q", "T")]["RW_export_share_among_active"]],
        ["US_export_share_QO", on["moments"][("Q", "O")]["US_export_share_among_active"], off["moments"][("Q", "O")]["US_export_share_among_active"]],
        ["RW_export_share_QO", on["moments"][("Q", "O")]["RW_export_share_among_active"], off["moments"][("Q", "O")]["RW_export_share_among_active"]],
        ["compliance_share_US_exporters_QT", on["moments"][("Q", "T")]["compliance_share_among_US_exporters"], 0.0],
        ["upgrade_share_active_QT", on["moments"][("Q", "T")]["upgrade_share_among_active"], off["moments"][("Q", "T")]["upgrade_share_among_active"]],
        ["Q_US_total", summary["trade_on"]["Q_US_total"], summary["trade_off"]["Q_US_total"]],
        ["Q_RW_total", summary["trade_on"]["Q_RW_total"], summary["trade_off"]["Q_RW_total"]],
        ["Q_US_T", summary["trade_on"]["Q_US_T"], summary["trade_off"]["Q_US_T"]],
        ["Q_RW_T", summary["trade_on"]["Q_RW_T"], summary["trade_off"]["Q_RW_T"]],
        ["Q_US_O", summary["trade_on"]["Q_US_O"], summary["trade_off"]["Q_US_O"]],
        ["Q_RW_O", summary["trade_on"]["Q_RW_O"], summary["trade_off"]["Q_RW_O"]],
        ["US_exporters_total", summary["trade_on"]["US_exporters_total"], summary["trade_off"]["US_exporters_total"]],
        ["RW_exporters_total", summary["trade_on"]["RW_exporters_total"], summary["trade_off"]["RW_exporters_total"]],
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def main() -> None:
    p0 = build_simple_params()

    stage1 = coordinate_stage1(p0)
    stage23 = calibrate_compliance_upgrade(p0, stage1)

    p = stage23["p"]
    on = stage23["on"]
    off = stage23["off"]
    trade_on = trade_summary(on, p)
    trade_off = trade_summary(off, p)
    same_type = m.summarize_qt_same_type_changes(on, off, p, disable_upgrade=False)

    summary = {
        "model": {
            "regions": p["regions"],
            "sectors": p["sectors"],
            "upgrading": "binary_textile_only",
            "roo_cost_formula": p["roo_cost_formula"],
            "roo_cost_scope": p["roo_cost_scope"],
            "use_admin_wedge": p["use_admin_wedge"],
            "textile_only_policy": True,
        },
        "targets": TARGETS,
        "stage1_best": {
            "state": stage1["state"],
            "loss": stage1["loss"],
            "participation_off_T": stage1["part_t"],
            "participation_off_O": stage1["part_o"],
        },
        "stage23_best": {
            "fC_mean_T": stage23["fC_mean_T"],
            "f_upgrade_T": stage23["f_upgrade_T"],
            "loss": stage23["loss"],
            "compliance_share_on_T": stage23["uptake_T"],
            "upgrade_share_on_T": stage23["upgrade_T"],
            "sigma_C_T": p["sigma_C"]["T"],
            "delta_T": p["delta"]["T"],
        },
        "params": {
            "f_dom_T": p["f_dom"][("Q", "T")],
            "f_export_US_T": p["f_export"][("Q", "US", "T")],
            "f_export_RW_T": p["f_export"][("Q", "RW", "T")],
            "f_dom_O": p["f_dom"][("Q", "O")],
            "f_export_US_O": p["f_export"][("Q", "US", "O")],
            "f_export_RW_O": p["f_export"][("Q", "RW", "O")],
            "f_entry_Q_T": p["f_entry"][("Q", "T")],
            "f_entry_Q_O": p["f_entry"][("Q", "O")],
            "fC_mean_T": p["fC_mean"]["T"],
            "sigma_C_T": p["sigma_C"]["T"],
            "f_upgrade_T": p["f_upgrade"]["T"],
            "delta_T": p["delta"]["T"],
            "t_mfn_T": p["t_mfn"]["T"],
            "t_mfn_O": p["t_mfn"]["O"],
        },
        "qiz_on_moments_T": on["moments"][("Q", "T")],
        "qiz_off_moments_T": off["moments"][("Q", "T")],
        "qiz_on_moments_O": on["moments"][("Q", "O")],
        "qiz_off_moments_O": off["moments"][("Q", "O")],
        "trade_on": trade_on,
        "trade_off": trade_off,
        "same_type_textile_changes": same_type,
        "welfare_on": float(on["welfare"]),
        "welfare_off": float(off["welfare"]),
        "welfare_pct": float(100.0 * (on["welfare"] / off["welfare"] - 1.0)),
        "Q_US_total_pct": float(100.0 * (trade_on["Q_US_total"] / max(trade_off["Q_US_total"], 1.0e-12) - 1.0)),
        "Q_RW_total_pct": float(100.0 * (trade_on["Q_RW_total"] / max(trade_off["Q_RW_total"], 1.0e-12) - 1.0)),
        "Q_US_T_pct": float(100.0 * (trade_on["Q_US_T"] / max(trade_off["Q_US_T"], 1.0e-12) - 1.0)),
        "Q_RW_T_pct": float(100.0 * (trade_on["Q_RW_T"] / max(trade_off["Q_RW_T"], 1.0e-12) - 1.0)),
        "US_exporters_total_pct": float(100.0 * (trade_on["US_exporters_total"] / max(trade_off["US_exporters_total"], 1.0e-12) - 1.0)),
        "RW_exporters_total_pct": float(100.0 * (trade_on["RW_exporters_total"] / max(trade_off["RW_exporters_total"], 1.0e-12) - 1.0)),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(on, off, summary)

    print(f"Saved summary: {OUT_JSON}")
    print(f"Saved QIZ comparison CSV: {OUT_CSV}")
    print(
        "Summary:",
        {
            "participation_off_T": stage1["part_t"],
            "participation_off_O": stage1["part_o"],
            "compliance_on_T": stage23["uptake_T"],
            "upgrade_on_T": stage23["upgrade_T"],
            "Q_US_T_pct": summary["Q_US_T_pct"],
            "Q_RW_T_pct": summary["Q_RW_T_pct"],
            "welfare_pct": summary["welfare_pct"],
        },
    )


if __name__ == "__main__":
    main()
