#!/usr/bin/env python3
"""
Grouping-2 multi-sector diagnostic with a common export fixed cost per sector.

Restriction:
- f_export[(Q,US,s)] = f_export[(Q,RW,s)] for each sector s

This keeps destination differences coming only from tariffs, iceberg costs,
and foreign expenditure shifters.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List

import qiz_model_ge as m
import simple_qiz_grouping2_gamma_crowdin as base
from run_presentation_analysis import summarize_gamma_mechanism, summarize_israeli_input_use


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "simple_qiz_grouping2_gamma_crowdin_common_export_cost_summary.json")

GRID_STAGE1 = {
    "f_dom": {
        "T": [0.7, 0.8, 0.9, 1.1],
        "S1": [0.5, 0.7, 0.9, 1.1],
        "S2": [0.3, 0.5, 0.7, 0.9, 1.1],
        "S3": [0.8, 1.0, 1.2, 1.4],
        "O": [0.6, 0.8, 1.0, 1.2],
    },
    "f_export_common": {
        "T": [8.0, 12.0, 18.0, 25.0, 35.0],
        "S1": [2.0, 4.0, 6.0, 8.0, 12.0, 20.0],
        "S2": [0.2, 0.5, 1.0, 2.0, 4.0, 6.0],
        "S3": [4.0, 6.0, 8.0, 12.0, 18.0],
        "O": [3.0, 4.0, 6.0, 8.0, 12.0],
    },
}


def build_params() -> Dict[str, Any]:
    p = base.build_params()
    for s in base.SECTORS:
        common = 0.5 * (
            float(p["f_export"][("Q", "US", s)]) + float(p["f_export"][("Q", "RW", s)])
        )
        p["f_export"][("Q", "US", s)] = common
        p["f_export"][("Q", "RW", s)] = common
    return p


def set_fixed_costs(p: Dict[str, Any], state: Dict[str, Dict[str, float]]) -> None:
    for s in base.SECTORS:
        p["f_dom"][("Q", s)] = float(state["f_dom"][s])
        common = float(state["f_export_common"][s])
        p["f_export"][("Q", "US", s)] = common
        p["f_export"][("Q", "RW", s)] = common


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
            initial_state=warm or base.default_initial_state(p),
            verbose=False,
        )
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc), "loss": float("inf")}

    parts = {s: base.participation_from_solution(sol, s) for s in base.SECTORS}
    return {"ok": True, "p": p, "sol": sol, "parts": parts, "loss": base.stage1_loss(parts, targets), "state": deepcopy(state)}


def coordinate_stage1(base_p: Dict[str, Any], targets: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    current = {
        "f_dom": {s: float(base_p["f_dom"][("Q", s)]) for s in base.SECTORS},
        "f_export_common": {s: float(base_p["f_export"][("Q", "US", s)]) for s in base.SECTORS},
    }
    best = run_off_equilibrium(base_p, current, targets)
    if not best.get("ok", False):
        raise RuntimeError(f"Initial stage1 solve failed: {best.get('error')}")

    for round_idx in range(3):
        improved = False
        warm = best["sol"]
        for block_name in ["f_dom", "f_export_common"]:
            for s in base.SECTORS:
                local_best = best
                for value in GRID_STAGE1[block_name][s]:
                    trial = deepcopy(current)
                    trial[block_name][s] = float(value)
                    rec = run_off_equilibrium(base_p, trial, targets, warm=warm)
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
    for fC in base.GRID_STAGE2_FC:
        p = deepcopy(base_p)
        set_fixed_costs(p, stage1["state"])
        p["fC_mean"] = {s: float(fC) for s in base.SECTORS}
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
    baseline_gamma = {s: float(p["gamma"][s]) for s in base.SECTORS}
    state = baseline_on
    rows: List[Dict[str, Any]] = []
    mechanism_rows: List[Dict[str, Any]] = []

    for gamma_pct in base.GAMMA_GRID_PCT:
        gamma_override = {s: gamma_pct / 100.0 for s in base.SECTORS}
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
            "exports_total_US": float(tr["exports"][("Q", "US")]),
            "exports_total_RW": float(tr["exports"][("Q", "RW")]),
        }
        for s in base.SECTORS:
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
    targets = base.load_grouping2_targets()
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
            "common_export_fixed_cost_across_destinations": True,
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
        "baseline_qiz_on_moments": {
            s: {k: float(v) for k, v in stage2["on"]["moments"][("Q", s)].items()} for s in base.SECTORS
        },
        "baseline_qiz_off_moments": {
            s: {k: float(v) for k, v in stage2["off"]["moments"][("Q", s)].items()} for s in base.SECTORS
        },
        "gamma_path": path["gamma_rows"],
        "gamma_mechanism": path["mechanism_rows"],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary:", OUT_JSON)
    print("Baseline compliance among US exporters by sector:", {
        s: stage2["on"]["moments"][("Q", s)]["compliance_share_among_US_exporters"] for s in base.SECTORS
    })


if __name__ == "__main__":
    main()
