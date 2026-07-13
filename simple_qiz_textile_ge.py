#!/usr/bin/env python3
"""
Simplified QIZ general-equilibrium experiment:

- one region: Q
- one sector: T (textiles/apparel)
- no upgrading
- no interregional labor mobility
- endogenous entry and export participation remain
- QIZ works only through the US tariff/compliance margin

The goal is diagnostic, not a replacement for the full model.
We calibrate a minimal set of fixed costs to textile participation moments and
the QIZ compliance rate, then compare QIZ on vs off.
"""

from __future__ import annotations

import csv
import json
import math
import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np

import qiz_model_ge as m


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "simple_qiz_textile_summary.json")
OUT_CSV = os.path.join(ROOT, "simple_qiz_textile_qiz_vs_noqiz.csv")

TARGETS = {
    "dom_only": 0.8927,
    "exp_US": 0.0575,
    "exp_RW": 0.0690,
    "compliance_rate": 0.321,
}

GRID_STAGE1 = {
    "f_dom_T": [0.8, 1.0, 1.3, 1.6, 2.0],
    "f_export_US_T": [3.0, 5.0, 7.5, 10.0, 14.0, 20.0],
    "f_export_RW_T": [3.0, 5.0, 7.5, 10.0, 14.0, 20.0],
}

GRID_STAGE2 = {
    "fC_mean_T": [0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0],
}


def build_simple_params() -> Dict[str, Any]:
    p = deepcopy(m.params_defensible())

    p["regions"] = ["Q"]
    p["sectors"] = ["T"]
    p["dests"] = ["EG", "US", "RW"]

    p["sigma"] = {"T": p["sigma"]["T"]}
    p["beta"] = {"T": 1.0}
    p["alpha"] = {"T": p["alpha"]["T"]}
    p["phi_min"] = {"T": p["phi_min"]["T"]}
    p["theta"] = {"T": p["theta"]["T"]}

    # Strip the model down to the tariff/compliance mechanism.
    p["delta"] = {"T": 1.0}
    p["upgrade_requires_US"] = False
    p["upgrade_mode"] = "binary"
    p["f_upgrade"] = {"T": 1.0e6}
    p["upgrade_psi"] = {"T": 0.0}
    p["upgrade_psi_comp"] = {"T": 0.0}
    p["upgrade_cost_fixed"] = {"T": 0.0}
    p["upgrade_cost_lin"] = {"T": 0.0}
    p["upgrade_cost_quad"] = {"T": 1.0}
    p["upgrade_intensity_max"] = {"T": 0.0}

    # Keep the cleaner ROO implementation, but strip the extra admin extension.
    p["roo_cost_formula"] = "normalized"
    p["roo_cost_scope"] = "US_only"
    p["use_admin_wedge"] = False
    p["xi_admin"] = {"T": 0.0}

    p["t_mfn"] = {"T": p["t_mfn"]["T"]}
    p["gamma"] = {"T": p["gamma"]["T"]}
    p["p_rw"] = {"T": p["p_rw"]["T"]}
    p["p_il"] = {"T": p["p_il"]["T"]}

    p["d_iceberg"] = {
        ("Q", "EG", "T"): p["d_iceberg"][("Q", "EG", "T")],
        ("Q", "US", "T"): p["d_iceberg"][("Q", "US", "T")],
        ("Q", "RW", "T"): p["d_iceberg"][("Q", "RW", "T")],
    }

    p["f_dom"] = {("Q", "T"): 1.0}
    p["f_export"] = {
        ("Q", "US", "T"): 8.0,
        ("Q", "RW", "T"): 8.0,
    }
    # Normalize entry cost in the stripped-down model.
    p["f_entry"] = {("Q", "T"): 1.0}

    p["fC_mean"] = {"T": 2.0}
    p["sigma_C"] = {"T": 0.35}

    # One region => fixed labor supply, no regional reallocation.
    p["L_total"] = 1.0
    p["L_Q_share"] = 1.0
    p["kappa"] = 1.0

    p["e_ratio_foreign"] = {
        ("US", "T"): p["e_ratio_foreign"][("US", "T")],
        ("RW", "T"): p["e_ratio_foreign"][("RW", "T")],
    }
    p["E_foreign"] = {("US", "T"): 1.0, ("RW", "T"): 1.0}
    p["P_foreign"] = {("US", "T"): 1.0, ("RW", "T"): 1.0}

    # Use moderate resolution; the stripped model is cheap to solve.
    p["n_phi"] = 100
    p["n_eps"] = 7
    p["goods_max_iter"] = 200
    p["outer_max_iter"] = 400
    p["outer_tol"] = 1.0e-4
    p["outer_cycle_tol"] = 5.0e-3
    return p


def participation_from_solution(sol: Dict[str, Any]) -> Dict[str, float]:
    mom = sol["moments"][("Q", "T")]
    return {
        "dom_only": float(mom["domestic_only_share_among_active"]),
        "exp_US": float(mom["US_export_share_among_active"]),
        "exp_RW": float(mom["RW_export_share_among_active"]),
        "active_share": float(mom["active_share"]),
    }


def stage1_loss(part: Dict[str, float]) -> float:
    return (
        ((part["dom_only"] - TARGETS["dom_only"]) / 0.03) ** 2
        + ((part["exp_US"] - TARGETS["exp_US"]) / 0.02) ** 2
        + ((part["exp_RW"] - TARGETS["exp_RW"]) / 0.02) ** 2
    )


def stage2_loss(uptake: float) -> float:
    return ((uptake - TARGETS["compliance_rate"]) / 0.04) ** 2


def set_stage1_costs(p: Dict[str, Any], state: Dict[str, float]) -> None:
    p["f_dom"][("Q", "T")] = state["f_dom_T"]
    p["f_export"][("Q", "US", "T")] = state["f_export_US_T"]
    p["f_export"][("Q", "RW", "T")] = state["f_export_RW_T"]


def run_off_equilibrium(base_p: Dict[str, Any], state: Dict[str, float], warm: Dict[str, Any] | None = None) -> Dict[str, Any]:
    p = deepcopy(base_p)
    set_stage1_costs(p, state)
    sol = m.solve_equilibrium(p, qiz_on=False, disable_upgrade=True, initial_state=warm, verbose=False)
    part = participation_from_solution(sol)
    return {"p": p, "sol": sol, "part": part, "loss": stage1_loss(part), "state": dict(state)}


def coordinate_stage1(base_p: Dict[str, Any]) -> Dict[str, Any]:
    current = {"f_dom_T": 1.0, "f_export_US_T": 8.0, "f_export_RW_T": 8.0}
    best = run_off_equilibrium(base_p, current)

    for round_idx in range(3):
        improved = False
        warm = best["sol"]
        for key, values in GRID_STAGE1.items():
            local_best = best
            for value in values:
                candidate = dict(current)
                candidate[key] = float(value)
                rec = run_off_equilibrium(base_p, candidate, warm=warm)
                if rec["loss"] < local_best["loss"] - 1.0e-12:
                    local_best = rec
            if local_best["loss"] < best["loss"] - 1.0e-12:
                best = local_best
                current = dict(best["state"])
                warm = best["sol"]
                improved = True
                print(
                    f"stage1 round {round_idx + 1} {key}: "
                    f"loss={best['loss']:.4f} state={best['state']} part={best['part']}"
                )
        if not improved:
            break
    return best


def calibrate_compliance(base_p: Dict[str, Any], stage1: Dict[str, Any]) -> Dict[str, Any]:
    best = None
    warm_off = stage1["sol"]
    for fC in GRID_STAGE2["fC_mean_T"]:
        p = deepcopy(base_p)
        set_stage1_costs(p, stage1["state"])
        p["fC_mean"]["T"] = float(fC)
        on = m.solve_equilibrium(p, qiz_on=True, disable_upgrade=True, initial_state=warm_off, verbose=False)
        off = m.solve_equilibrium(p, qiz_on=False, disable_upgrade=True, initial_state=on, verbose=False)
        uptake = float(on["moments"][("Q", "T")]["compliance_share_among_US_exporters"])
        rec = {
            "p": p,
            "on": on,
            "off": off,
            "fC_mean_T": float(fC),
            "uptake": uptake,
            "loss": stage2_loss(uptake),
        }
        if best is None or rec["loss"] < best["loss"] - 1.0e-12:
            best = rec
            print(
                f"stage2 fC_mean_T={fC:.3f}: "
                f"loss={rec['loss']:.4f} uptake={uptake:.4f}"
            )
    if best is None:
        raise RuntimeError("Compliance calibration failed.")
    return best


def trade_summary(sol: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, float]:
    tr = m.summarize_trade(sol, p)
    return {
        "US_exports": float(tr["exports"][("Q", "US")]),
        "RW_exports": float(tr["exports"][("Q", "RW")]),
        "US_exporters": float(tr["exporter_masses"][("Q", "US")]),
        "RW_exporters": float(tr["exporter_masses"][("Q", "RW")]),
        "avg_US_rev_per_exporter": float(tr["avg_revenue_per_exporter"][("Q", "US")]),
        "avg_RW_rev_per_exporter": float(tr["avg_revenue_per_exporter"][("Q", "RW")]),
    }


def write_csv(on: Dict[str, Any], off: Dict[str, Any], summary: Dict[str, Any]) -> None:
    rows = [
        ["metric", "qiz_on", "qiz_off"],
        ["welfare", summary["welfare_on"], summary["welfare_off"]],
        ["price_index", on["goods"]["P_EG"], off["goods"]["P_EG"]],
        ["entry_mass_QT", on["M"][("Q", "T")], off["M"][("Q", "T")]],
        ["active_share_QT", on["moments"][("Q", "T")]["active_share"], off["moments"][("Q", "T")]["active_share"]],
        ["domestic_only_share_QT", on["moments"][("Q", "T")]["domestic_only_share_among_active"], off["moments"][("Q", "T")]["domestic_only_share_among_active"]],
        ["US_export_share_QT", on["moments"][("Q", "T")]["US_export_share_among_active"], off["moments"][("Q", "T")]["US_export_share_among_active"]],
        ["RW_export_share_QT", on["moments"][("Q", "T")]["RW_export_share_among_active"], off["moments"][("Q", "T")]["RW_export_share_among_active"]],
        ["compliance_share_US_exporters_QT", on["moments"][("Q", "T")]["compliance_share_among_US_exporters"], 0.0],
        ["US_exports", summary["trade_on"]["US_exports"], summary["trade_off"]["US_exports"]],
        ["RW_exports", summary["trade_on"]["RW_exports"], summary["trade_off"]["RW_exports"]],
        ["US_exporters", summary["trade_on"]["US_exporters"], summary["trade_off"]["US_exporters"]],
        ["RW_exporters", summary["trade_on"]["RW_exporters"], summary["trade_off"]["RW_exporters"]],
        ["avg_US_rev_per_exporter", summary["trade_on"]["avg_US_rev_per_exporter"], summary["trade_off"]["avg_US_rev_per_exporter"]],
        ["avg_RW_rev_per_exporter", summary["trade_on"]["avg_RW_rev_per_exporter"], summary["trade_off"]["avg_RW_rev_per_exporter"]],
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def main() -> None:
    p0 = build_simple_params()

    stage1 = coordinate_stage1(p0)
    stage2 = calibrate_compliance(p0, stage1)

    on = stage2["on"]
    off = stage2["off"]
    p = stage2["p"]
    trade_on = trade_summary(on, p)
    trade_off = trade_summary(off, p)

    summary = {
        "model": {
            "regions": p["regions"],
            "sectors": p["sectors"],
            "upgrading": "off",
            "roo_cost_formula": p["roo_cost_formula"],
            "roo_cost_scope": p["roo_cost_scope"],
            "use_admin_wedge": p["use_admin_wedge"],
            "entry_cost_normalization": p["f_entry"][("Q", "T")],
        },
        "targets": TARGETS,
        "stage1_best": {
            "state": stage1["state"],
            "loss": stage1["loss"],
            "participation_off": stage1["part"],
        },
        "stage2_best": {
            "fC_mean_T": stage2["fC_mean_T"],
            "loss": stage2["loss"],
            "compliance_share_on": stage2["uptake"],
            "sigma_C_T": p["sigma_C"]["T"],
        },
        "params": {
            "f_dom_T": p["f_dom"][("Q", "T")],
            "f_export_US_T": p["f_export"][("Q", "US", "T")],
            "f_export_RW_T": p["f_export"][("Q", "RW", "T")],
            "f_entry_Q_T": p["f_entry"][("Q", "T")],
            "fC_mean_T": p["fC_mean"]["T"],
            "sigma_C_T": p["sigma_C"]["T"],
            "t_mfn_T": p["t_mfn"]["T"],
            "gamma_T": p["gamma"]["T"],
        },
        "qiz_on_moments": on["moments"][("Q", "T")],
        "qiz_off_moments": off["moments"][("Q", "T")],
        "trade_on": trade_on,
        "trade_off": trade_off,
        "welfare_on": float(on["welfare"]),
        "welfare_off": float(off["welfare"]),
        "welfare_pct": float(100.0 * (on["welfare"] / off["welfare"] - 1.0)),
        "US_exports_pct": float(100.0 * (trade_on["US_exports"] / max(trade_off["US_exports"], 1.0e-12) - 1.0)),
        "RW_exports_pct": float(100.0 * (trade_on["RW_exports"] / max(trade_off["RW_exports"], 1.0e-12) - 1.0)),
        "US_exporters_pct": float(100.0 * (trade_on["US_exporters"] / max(trade_off["US_exporters"], 1.0e-12) - 1.0)),
        "RW_exporters_pct": float(100.0 * (trade_on["RW_exporters"] / max(trade_off["RW_exporters"], 1.0e-12) - 1.0)),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(on, off, summary)

    print(f"Saved summary: {OUT_JSON}")
    print(f"Saved QIZ comparison CSV: {OUT_CSV}")
    print(
        "Summary:",
        {
            "participation_off": stage1["part"],
            "compliance_on": stage2["uptake"],
            "US_exports_pct": summary["US_exports_pct"],
            "RW_exports_pct": summary["RW_exports_pct"],
            "welfare_pct": summary["welfare_pct"],
        },
    )


if __name__ == "__main__":
    main()
