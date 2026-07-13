#!/usr/bin/env python3
"""
Grouping-2 multi-sector diagnostic for QIZ crowd-in when the Israeli-content
requirement gamma is lowered.

Design:
- one region: Q
- five sectors: T, S1, S2, S3, O
- no upgrading (to isolate the compliance-selection mechanism)
- endogenous entry and export participation remain
- common compliance technology across sectors

This script calibrates off-policy participation to grouping_2 targets, then
calibrates a common compliance cost to match the textile compliance rate.
Finally, it runs a gamma path and reports which sectors are crowded into QIZ use.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List

import qiz_model_ge as m
from run_presentation_analysis import summarize_gamma_mechanism, summarize_israeli_input_use


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "simple_qiz_grouping2_gamma_crowdin_summary.json")

SECTORS = ["T", "S1", "S2", "S3", "O"]

GRID_STAGE1 = {
    "f_dom": {
        "T": [0.7, 0.8, 0.9, 1.1],
        "S1": [0.5, 0.7, 0.9, 1.1],
        "S2": [0.3, 0.5, 0.7, 0.9, 1.1],
        "S3": [0.8, 1.0, 1.2, 1.4],
        "O": [0.6, 0.8, 1.0, 1.2],
    },
    "f_export_US": {
        "T": [8.0, 12.0, 18.0, 25.0],
        "S1": [0.8, 1.0, 1.5, 2.0, 4.0, 8.0, 12.0, 20.0],
        "S2": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
        "S3": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0],
        "O": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    },
    "f_export_RW": {
        "T": [12.0, 20.0, 35.0, 50.0],
        "S1": [4.0, 6.0, 9.0, 12.0],
        "S2": [1.0, 2.0, 4.0, 6.0, 10.0, 14.0],
        "S3": [8.0, 12.0, 18.0, 24.0],
        "O": [4.0, 6.0, 9.0, 12.0],
    },
}

GRID_STAGE2_FC = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0]
GAMMA_GRID_PCT = [10.5, 8.0, 5.0, 2.0, 0.0]


def load_grouping2_targets() -> Dict[str, Any]:
    with open(os.path.join(ROOT, "params_estimated.json"), "r", encoding="utf-8") as f:
        p = json.load(f)
    gp = p["export_participation"]["grouping_2"]
    return {s: {"dom_only": gp[s]["dom_only"], "exp_US": gp[s]["exp_US"], "exp_RW": gp[s]["exp_RW"]} for s in SECTORS}


def load_grouping2_param_block() -> Dict[str, Dict[str, float]]:
    with open(os.path.join(ROOT, "params_estimated.json"), "r", encoding="utf-8") as f:
        pe = json.load(f)

    return {
        "beta": pe["beta_s"]["grouping_2"]["beta"],
        "alpha": pe["alpha_s"]["grouping_2"]["alpha"],
        "theta": pe["theta_s"]["grouping_2"]["theta"],
        "sigma": pe["theta_s"]["sigma_used"],
        "t_mfn": {s: pe["mfn_tariff"]["grouping_2"]["tau"][s] - 1.0 for s in SECTORS},
        "gamma": pe["gamma_s"]["grouping_2"],
        "p_il": pe["price_wedge"]["grouping_2"],
    }


def load_grouping2_absorption() -> Dict[str, Dict[str, float]]:
    with open(os.path.join(ROOT, "data_calibration", "absorption_2004.json"), "r", encoding="utf-8") as f:
        ab = json.load(f)["grouping_2"]
    return ab


def load_grouping2_trade_costs() -> Dict[str, Dict[str, float]]:
    with open(os.path.join(ROOT, "trade_costs.json"), "r", encoding="utf-8") as f:
        tc = json.load(f)["grouping_2"]["data"]
    return tc


def build_params() -> Dict[str, Any]:
    block = load_grouping2_param_block()
    ab = load_grouping2_absorption()
    tc = load_grouping2_trade_costs()
    p = deepcopy(m.params_defensible())

    p["regions"] = ["Q"]
    p["sectors"] = list(SECTORS)
    p["dests"] = ["EG", "US", "RW"]

    p["sigma"] = {s: float(block["sigma"][s]) for s in SECTORS}
    p["beta"] = {s: float(block["beta"][s]) for s in SECTORS}
    p["alpha"] = {s: float(block["alpha"][s]) for s in SECTORS}
    p["phi_min"] = {s: 1.0 for s in SECTORS}
    p["theta"] = {s: float(block["theta"][s]) for s in SECTORS}

    # Shut down upgrading to isolate compliance crowd-in.
    p["delta"] = {s: 1.0 for s in SECTORS}
    p["upgrade_requires_US"] = False
    p["upgrade_mode"] = "binary"
    p["f_upgrade"] = {s: 1.0e6 for s in SECTORS}
    p["upgrade_psi"] = {s: 0.0 for s in SECTORS}
    p["upgrade_psi_comp"] = {s: 0.0 for s in SECTORS}
    p["upgrade_cost_fixed"] = {s: 0.0 for s in SECTORS}
    p["upgrade_cost_lin"] = {s: 0.0 for s in SECTORS}
    p["upgrade_cost_quad"] = {s: 1.0 for s in SECTORS}
    p["upgrade_intensity_max"] = {s: 0.0 for s in SECTORS}

    p["roo_cost_formula"] = "normalized"
    p["roo_cost_scope"] = "US_only"
    p["use_admin_wedge"] = False
    p["xi_admin"] = {s: 0.0 for s in SECTORS}

    p["t_mfn"] = {s: float(block["t_mfn"][s]) for s in SECTORS}
    p["gamma"] = {s: float(block["gamma"][s]) for s in SECTORS}
    p["p_rw"] = {s: 1.0 for s in SECTORS}
    p["p_il"] = {s: float(block["p_il"][s]) for s in SECTORS}

    p["d_iceberg"] = {}
    for s in SECTORS:
        p["d_iceberg"][("Q", "EG", s)] = 1.0
        p["d_iceberg"][("Q", "US", s)] = float(tc[s]["trade_costs"]["tau_EGY_US"])
        p["d_iceberg"][("Q", "RW", s)] = float(tc[s]["trade_costs"]["tau_EGY_RoW"])

    # Stage-1 starting guesses for fixed costs.
    p["f_dom"] = {
        ("Q", "T"): 0.8,
        ("Q", "S1"): 0.7,
        ("Q", "S2"): 0.9,
        ("Q", "S3"): 1.0,
        ("Q", "O"): 0.8,
    }
    p["f_export"] = {
        ("Q", "US", "T"): 12.0,
        ("Q", "RW", "T"): 20.0,
        ("Q", "US", "S1"): 20.0,
        ("Q", "RW", "S1"): 6.0,
        ("Q", "US", "S2"): 20.0,
        ("Q", "RW", "S2"): 10.0,
        ("Q", "US", "S3"): 12.0,
        ("Q", "RW", "S3"): 12.0,
        ("Q", "US", "O"): 10.0,
        ("Q", "RW", "O"): 6.0,
    }
    p["f_entry"] = {("Q", s): 1.0 for s in SECTORS}

    # Common compliance technology across sectors.
    p["fC_mean"] = {s: 12.0 for s in SECTORS}
    p["sigma_C"] = {s: 0.35 for s in SECTORS}

    # One-region diagnostic.
    p["L_total"] = 1.0
    p["L_Q_share"] = 1.0
    p["kappa"] = 1.0

    p["e_ratio_foreign"] = {}
    for s in SECTORS:
        p["e_ratio_foreign"][("US", s)] = ab[s]["E_US"] / ab[s]["E_EGY"]
        p["e_ratio_foreign"][("RW", s)] = ab[s]["E_RoW"] / ab[s]["E_EGY"]
    p["E_foreign"] = {k: 1.0 for k in p["e_ratio_foreign"]}
    p["P_foreign"] = {(j, s): 1.0 for (j, s) in p["e_ratio_foreign"]}

    p["n_phi"] = 90
    p["n_eps"] = 7
    p["goods_max_iter"] = 250
    p["outer_max_iter"] = 420
    p["outer_tol"] = 1.0e-4
    p["outer_cycle_tol"] = 5.0e-3
    return p


def default_initial_state(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "w": {r: 1.0 for r in p["regions"]},
        "M": {(r, s): 0.01 for r in p["regions"] for s in p["sectors"]},
        "goods": {"P_EG_s": {s: 1.0 for s in p["sectors"]}},
    }


def participation_from_solution(sol: Dict[str, Any], sector: str) -> Dict[str, float]:
    mom = sol["moments"][("Q", sector)]
    return {
        "dom_only": float(mom["domestic_only_share_among_active"]),
        "exp_US": float(mom["US_export_share_among_active"]),
        "exp_RW": float(mom["RW_export_share_among_active"]),
        "active_share": float(mom["active_share"]),
    }


def set_fixed_costs(p: Dict[str, Any], state: Dict[str, Dict[str, float]]) -> None:
    for s in SECTORS:
        p["f_dom"][("Q", s)] = float(state["f_dom"][s])
        p["f_export"][("Q", "US", s)] = float(state["f_export_US"][s])
        p["f_export"][("Q", "RW", s)] = float(state["f_export_RW"][s])


def stage1_loss(parts: Dict[str, Dict[str, float]], targets: Dict[str, Dict[str, float]]) -> float:
    loss = 0.0
    for s in SECTORS:
        loss += ((parts[s]["dom_only"] - targets[s]["dom_only"]) / 0.04) ** 2
        loss += ((parts[s]["exp_US"] - targets[s]["exp_US"]) / 0.02) ** 2
        loss += ((parts[s]["exp_RW"] - targets[s]["exp_RW"]) / 0.03) ** 2
    return loss


def run_off_equilibrium(
    base_p: Dict[str, Any],
    state: Dict[str, Dict[str, float]],
    targets: Dict[str, Dict[str, float]],
    warm: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    p = deepcopy(base_p)
    set_fixed_costs(p, state)
    try:
        sol = m.solve_equilibrium(
            p,
            qiz_on=False,
            disable_upgrade=True,
            initial_state=warm or default_initial_state(p),
            verbose=False,
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc), "loss": float("inf")}

    parts = {s: participation_from_solution(sol, s) for s in SECTORS}
    return {"ok": True, "p": p, "sol": sol, "parts": parts, "loss": stage1_loss(parts, targets), "state": deepcopy(state)}


def coordinate_stage1(base_p: Dict[str, Any], targets: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    current = {
        "f_dom": {s: float(base_p["f_dom"][("Q", s)]) for s in SECTORS},
        "f_export_US": {s: float(base_p["f_export"][("Q", "US", s)]) for s in SECTORS},
        "f_export_RW": {s: float(base_p["f_export"][("Q", "RW", s)]) for s in SECTORS},
    }
    best = run_off_equilibrium(base_p, current, targets)
    if not best.get("ok", False):
        raise RuntimeError(f"Initial stage1 solve failed: {best.get('error')}")

    for round_idx in range(3):
        improved = False
        warm = best["sol"]
        for block_name in ["f_dom", "f_export_US", "f_export_RW"]:
            for s in SECTORS:
                local_best = best
                for value in GRID_STAGE1[block_name][s]:
                    cand = deepcopy(current)
                    cand[block_name][s] = float(value)
                    rec = run_off_equilibrium(base_p, cand, targets, warm=warm)
                    if rec.get("ok", False) and rec["loss"] < local_best["loss"] - 1.0e-12:
                        local_best = rec
                if local_best["loss"] < best["loss"] - 1.0e-12:
                    best = local_best
                    current = deepcopy(best["state"])
                    warm = best["sol"]
                    improved = True
                    print(
                        f"stage1 round {round_idx + 1} {block_name}[{s}]: "
                        f"loss={best['loss']:.4f} part={best['parts'][s]}"
                    )
        if not improved:
            break
    return best


def calibrate_common_compliance_cost(base_p: Dict[str, Any], stage1: Dict[str, Any]) -> Dict[str, Any]:
    best = None
    warm = stage1["sol"]
    for fC in GRID_STAGE2_FC:
        p = deepcopy(base_p)
        set_fixed_costs(p, stage1["state"])
        p["fC_mean"] = {s: float(fC) for s in SECTORS}
        try:
            on = m.solve_equilibrium(p, qiz_on=True, disable_upgrade=True, initial_state=warm, verbose=False)
            off = m.solve_equilibrium(p, qiz_on=False, disable_upgrade=True, initial_state=on, verbose=False)
        except RuntimeError:
            continue
        uptake_t = float(on["moments"][("Q", "T")]["compliance_share_among_US_exporters"])
        loss = ((uptake_t - 0.321) / 0.05) ** 2
        rec = {"p": p, "on": on, "off": off, "fC_common": float(fC), "uptake_T": uptake_t, "loss": loss}
        if best is None or rec["loss"] < best["loss"] - 1.0e-12:
            best = rec
            print(f"stage2 fC_common={fC:.3f}: loss={loss:.4f} uptake_T={uptake_t:.4f}")
    if best is None:
        raise RuntimeError("Common compliance calibration failed.")
    return best


def gamma_path_summary(p: Dict[str, Any], baseline_on: Dict[str, Any]) -> Dict[str, Any]:
    baseline_gamma = {s: float(p["gamma"][s]) for s in SECTORS}
    state = baseline_on
    rows: List[Dict[str, Any]] = []
    mechanism_rows: List[Dict[str, Any]] = []

    for gamma_pct in GAMMA_GRID_PCT:
        gamma_override = {s: gamma_pct / 100.0 for s in SECTORS}
        sol = m.solve_equilibrium(
            p,
            qiz_on=True,
            gamma_override=gamma_override,
            disable_upgrade=True,
            initial_state=state,
            verbose=False,
        )
        state = sol
        tr = m.summarize_trade(sol, p)
        sec = m.summarize_trade_by_sector(sol, p)
        il = summarize_israeli_input_use(sol, p)
        mech = summarize_gamma_mechanism(
            baseline_sol=baseline_on,
            current_sol=sol,
            p=p,
            baseline_gamma=baseline_gamma,
            current_gamma=gamma_override,
        )
        mechanism_rows.extend([{**row, "gamma_pct": float(gamma_pct)} for row in mech.to_dict(orient="records")])

        row: Dict[str, Any] = {
            "gamma_pct": float(gamma_pct),
            "welfare": float(sol["welfare"]),
            "real_wage_Q": float(sol["w"]["Q"] / sol["goods"]["P_EG"]),
            "israeli_inputs_total": float(il["total"]),
            "exports_total_US": float(sum(tr["exports"][("Q", "US")] for _ in [0])),
            "exports_total_RW": float(sum(tr["exports"][("Q", "RW")] for _ in [0])),
        }
        for s in SECTORS:
            mom = sol["moments"][("Q", s)]
            row[f"comp_share_active_{s}"] = float(mom["compliance_share_among_active"])
            row[f"comp_share_us_exporters_{s}"] = float(mom["compliance_share_among_US_exporters"])
            row[f"us_share_active_{s}"] = float(mom["US_export_share_among_active"])
            row[f"rw_share_active_{s}"] = float(mom["RW_export_share_among_active"])
            row[f"exports_US_{s}"] = float(sec[("Q", s, "US")])
            row[f"exports_RW_{s}"] = float(sec[("Q", s, "RW")])
            row[f"israeli_inputs_{s}"] = float(il.get(f"sector_{s}", 0.0))
        rows.append(row)

    return {"gamma_rows": rows, "mechanism_rows": mechanism_rows}


def main() -> None:
    targets = load_grouping2_targets()
    p0 = build_params()
    stage1 = coordinate_stage1(p0, targets)
    stage2 = calibrate_common_compliance_cost(p0, stage1)

    path = gamma_path_summary(stage2["p"], stage2["on"])

    summary = {
        "model": {
            "regions": stage2["p"]["regions"],
            "sectors": stage2["p"]["sectors"],
            "upgrading": "off",
            "roo_cost_formula": stage2["p"]["roo_cost_formula"],
            "roo_cost_scope": stage2["p"]["roo_cost_scope"],
            "use_admin_wedge": stage2["p"]["use_admin_wedge"],
            "common_compliance_cost_across_sectors": True,
        },
        "targets": targets,
        "stage1_best": {
            "state": stage1["state"],
            "loss": stage1["loss"],
            "participation_off": stage1["parts"],
        },
        "stage2_best": {
            "fC_common": stage2["fC_common"],
            "loss": stage2["loss"],
            "uptake_T": stage2["uptake_T"],
        },
        "baseline_qiz_on_moments": {s: stage2["on"]["moments"][("Q", s)] for s in SECTORS},
        "baseline_qiz_off_moments": {s: stage2["off"]["moments"][("Q", s)] for s in SECTORS},
        "gamma_path": path["gamma_rows"],
        "gamma_mechanism": path["mechanism_rows"],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary: {OUT_JSON}")
    print(
        "Baseline compliance among US exporters by sector:",
        {s: stage2['on']['moments'][('Q', s)]['compliance_share_among_US_exporters'] for s in SECTORS},
    )


if __name__ == "__main__":
    main()
