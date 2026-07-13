#!/usr/bin/env python3
"""
Grouping-2 multi-sector diagnostic with compliance-induced upgrading.

Design:
- one region: Q
- five sectors: T, S1, S2, S3, O
- endogenous entry and export participation
- QIZ compliance available in all sectors
- upgrading available in all sectors as a diagnostic extension

This extends the no-upgrade grouping_2 script by adding a smooth upgrading
margin. Because the observed 49.1% textile upgrading target is measured among
complying firms, calibration here targets upgrading among compliant textile
firms, not among all active firms.
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
OUT_JSON = os.path.join(ROOT, "simple_qiz_grouping2_upgrade_gamma_crowdin_summary.json")
BASE_STAGE1_JSON = os.path.join(ROOT, "simple_qiz_grouping2_gamma_crowdin_summary.json")

# Small hand-picked candidate set. Diagnostics showed the smooth-upgrade extension
# tends to an all-or-none upgrade decision among compliers under common costs,
# so an exhaustive grid was expensive without changing the qualitative result.
UPGRADE_CANDIDATES = [
    {"psi_comp_common": 0.020, "fC_common": 20.0, "upgrade_cost_fixed_common": 20.0, "upgrade_cost_quad_common": 4.0},
    {"psi_comp_common": 0.035, "fC_common": 35.0, "upgrade_cost_fixed_common": 20.0, "upgrade_cost_quad_common": 4.0},
    {"psi_comp_common": 0.040, "fC_common": 50.0, "upgrade_cost_fixed_common": 20.0, "upgrade_cost_quad_common": 4.0},
]

TARGETS = {
    "compliance_rate_T": 0.3207,
    "upgrade_rate_complying_T": 0.4910,
}


def build_params() -> Dict[str, Any]:
    p = base.build_params()

    # Turn on smooth upgrading. A complying firm can choose intensity u in [0, 2].
    # With psi_comp = 0.10, max productivity gain is exp(0.2) = 1.221, close to the
    # textile estimate delta_T = 1.178 used elsewhere in the project.
    p["upgrade_mode"] = "continuous"
    p["upgrade_requires_US"] = False
    p["delta"] = {s: 1.0 for s in base.SECTORS}
    p["f_upgrade"] = {s: 1.0e6 for s in base.SECTORS}
    p["upgrade_psi"] = {s: 0.0 for s in base.SECTORS}
    p["upgrade_psi_comp"] = {s: 0.035 for s in base.SECTORS}
    p["upgrade_cost_fixed"] = {s: 20.0 for s in base.SECTORS}
    p["upgrade_cost_lin"] = {s: 0.0 for s in base.SECTORS}
    p["upgrade_cost_quad"] = {s: 4.0 for s in base.SECTORS}
    p["upgrade_intensity_max"] = {s: 2.0 for s in base.SECTORS}
    p["n_phi"] = 70
    p["n_eps"] = 5
    p["goods_max_iter"] = 220
    p["outer_max_iter"] = 320
    p["outer_tol"] = 2.0e-4
    p["outer_cycle_tol"] = 7.5e-3
    return p


def get_stage1(base_p: Dict[str, Any], targets: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    if os.path.exists(BASE_STAGE1_JSON):
        with open(BASE_STAGE1_JSON, "r", encoding="utf-8") as f:
            cached = json.load(f)
        state = cached["stage1_best"]["state"]
        rec = base.run_off_equilibrium(base_p, state, targets)
        if rec.get("ok", False):
            print(f"Loaded cached stage1 state from {BASE_STAGE1_JSON}")
            return rec
    return base.coordinate_stage1(base_p, targets)


def compliance_upgrade_stats(
    sol: Dict[str, Any],
    p: Dict[str, Any],
    sector: str,
    gamma_override: Dict[str, float] | None = None,
) -> Dict[str, float]:
    phi_grid, w_phi = m.pareto_grid(p["phi_min"][sector], p["theta"][sector], p["n_phi"])
    eps_grid, w_eps = m.normal_grid(p["n_eps"], a=p["eps_std"])

    r = "Q"
    comp_mass = 0.0
    up_comp_mass = 0.0
    up_comp_intensity = 0.0

    for phi, w_phi_i in zip(phi_grid, w_phi):
        for eps, w_eps_i in zip(eps_grid, w_eps):
            wt = w_phi_i * w_eps_i
            best = m.firm_best(
                phi,
                eps,
                r,
                sector,
                sol["w"][r],
                sol["goods"]["P_EG_s"][sector],
                sol["goods"]["E_EG_s"][sector],
                p,
                gamma_override=gamma_override,
                qiz_on=sol.get("qiz_on", True),
                disable_upgrade=sol.get("disable_upgrade", False),
            )
            if not best["compliance"]:
                continue
            comp_mass += wt
            if best["upgrade"]:
                up_comp_mass += wt
            up_comp_intensity += wt * best.get("upgrade_intensity", 0.0)

    if comp_mass <= 1.0e-12:
        return {
            "upgrade_share_among_compliers": 0.0,
            "upgrade_intensity_among_compliers": 0.0,
            "complier_type_mass": 0.0,
        }

    return {
        "upgrade_share_among_compliers": float(up_comp_mass / comp_mass),
        "upgrade_intensity_among_compliers": float(up_comp_intensity / comp_mass),
        "complier_type_mass": float(comp_mass),
    }


def stage2_loss(uptake_t: float, upgrade_comp_t: float) -> float:
    return (
        ((uptake_t - TARGETS["compliance_rate_T"]) / 0.05) ** 2
        + ((upgrade_comp_t - TARGETS["upgrade_rate_complying_T"]) / 0.08) ** 2
    )


def calibrate_compliance_upgrade(base_p: Dict[str, Any], stage1: Dict[str, Any]) -> Dict[str, Any]:
    best = None
    warm = stage1["sol"]

    for cand in UPGRADE_CANDIDATES:
        p = deepcopy(base_p)
        base.set_fixed_costs(p, stage1["state"])
        p["fC_mean"] = {s: float(cand["fC_common"]) for s in base.SECTORS}
        p["upgrade_psi_comp"] = {s: float(cand["psi_comp_common"]) for s in base.SECTORS}
        p["upgrade_cost_fixed"] = {s: float(cand["upgrade_cost_fixed_common"]) for s in base.SECTORS}
        p["upgrade_cost_quad"] = {s: float(cand["upgrade_cost_quad_common"]) for s in base.SECTORS}
        try:
            on = m.solve_equilibrium(p, qiz_on=True, disable_upgrade=False, initial_state=warm, verbose=False)
            off = m.solve_equilibrium(p, qiz_on=False, disable_upgrade=False, initial_state=on, verbose=False)
        except RuntimeError:
            continue

        uptake_t = float(on["moments"][("Q", "T")]["compliance_share_among_US_exporters"])
        up_t = compliance_upgrade_stats(on, p, "T")
        upgrade_comp_t = float(up_t["upgrade_share_among_compliers"])
        rec = {
            "p": p,
            "on": on,
            "off": off,
            "psi_comp_common": float(cand["psi_comp_common"]),
            "fC_common": float(cand["fC_common"]),
            "upgrade_cost_fixed_common": float(cand["upgrade_cost_fixed_common"]),
            "upgrade_cost_quad_common": float(cand["upgrade_cost_quad_common"]),
            "uptake_T": uptake_t,
            "upgrade_complying_T": upgrade_comp_t,
            "loss": stage2_loss(uptake_t, upgrade_comp_t),
        }
        if best is None or rec["loss"] < best["loss"] - 1.0e-12:
            best = rec
            warm = on
            print(
                f"stage2 psi={cand['psi_comp_common']:.3f} fC={cand['fC_common']:.1f} "
                f"f_fixed={cand['upgrade_cost_fixed_common']:.1f} quad={cand['upgrade_cost_quad_common']:.1f}: "
                f"loss={rec['loss']:.4f} uptake_T={uptake_t:.4f} upgrade_comp_T={upgrade_comp_t:.4f}"
            )

    if best is None:
        raise RuntimeError("Compliance/upgrade calibration failed.")
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
            disable_upgrade=False,
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
            up = compliance_upgrade_stats(sol, p, s, gamma_override=gamma_override)
            row[f"comp_share_active_{s}"] = float(mom["compliance_share_among_active"])
            row[f"comp_share_us_exporters_{s}"] = float(mom["compliance_share_among_US_exporters"])
            row[f"upgrade_share_active_{s}"] = float(mom["upgrade_share_among_active"])
            row[f"upgrade_intensity_active_{s}"] = float(mom["upgrade_intensity_among_active"])
            row[f"upgrade_share_compliers_{s}"] = float(up["upgrade_share_among_compliers"])
            row[f"upgrade_intensity_compliers_{s}"] = float(up["upgrade_intensity_among_compliers"])
            row[f"us_share_active_{s}"] = float(mom["US_export_share_among_active"])
            row[f"rw_share_active_{s}"] = float(mom["RW_export_share_among_active"])
            row[f"exports_US_{s}"] = float(sec[("Q", s, "US")])
            row[f"exports_RW_{s}"] = float(sec[("Q", s, "RW")])
            row[f"israeli_inputs_{s}"] = float(il.get(f"sector_{s}", 0.0))
        rows.append(row)

    return {"gamma_rows": rows, "mechanism_rows": mechanism_rows}


def main() -> None:
    p0 = build_params()
    targets = base.load_grouping2_targets()
    stage1 = get_stage1(p0, targets)
    stage2 = calibrate_compliance_upgrade(p0, stage1)

    path = gamma_path_summary(stage2["p"], stage2["on"])

    baseline_upgrade_compliers = {
        s: compliance_upgrade_stats(stage2["on"], stage2["p"], s) for s in base.SECTORS
    }

    summary = {
        "model": {
            "regions": stage2["p"]["regions"],
            "sectors": stage2["p"]["sectors"],
            "upgrading": "continuous_compliance_bonus",
            "roo_cost_formula": stage2["p"]["roo_cost_formula"],
            "roo_cost_scope": stage2["p"]["roo_cost_scope"],
            "use_admin_wedge": stage2["p"]["use_admin_wedge"],
            "common_compliance_cost_across_sectors": True,
            "common_upgrade_cost_across_sectors": True,
            "upgrade_psi_comp_common": stage2["p"]["upgrade_psi_comp"]["T"],
            "upgrade_intensity_max_common": stage2["p"]["upgrade_intensity_max"]["T"],
        },
        "targets": {
            "participation_off": targets,
            "compliance_rate_T": TARGETS["compliance_rate_T"],
            "upgrade_rate_complying_T": TARGETS["upgrade_rate_complying_T"],
        },
        "stage1_best": {
            "state": stage1["state"],
            "loss": stage1["loss"],
            "participation_off": stage1["parts"],
        },
        "stage2_best": {
            "psi_comp_common": stage2["psi_comp_common"],
            "fC_common": stage2["fC_common"],
            "upgrade_cost_fixed_common": stage2["upgrade_cost_fixed_common"],
            "upgrade_cost_quad_common": stage2["upgrade_cost_quad_common"],
            "loss": stage2["loss"],
            "uptake_T": stage2["uptake_T"],
            "upgrade_complying_T": stage2["upgrade_complying_T"],
        },
        "baseline_qiz_on_moments": {
            s: {k: float(v) for k, v in stage2["on"]["moments"][("Q", s)].items()} for s in base.SECTORS
        },
        "baseline_qiz_off_moments": {
            s: {k: float(v) for k, v in stage2["off"]["moments"][("Q", s)].items()} for s in base.SECTORS
        },
        "baseline_upgrade_among_compliers": baseline_upgrade_compliers,
        "gamma_path": path["gamma_rows"],
        "gamma_mechanism": path["mechanism_rows"],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary:", OUT_JSON)
    print(
        "Baseline textile moments:",
        {
            "compliance_USexp_T": stage2["uptake_T"],
            "upgrade_complying_T": stage2["upgrade_complying_T"],
        },
    )


if __name__ == "__main__":
    main()
