#!/usr/bin/env python3
"""
Joint calibration of the QIZ model around observed pre-QIZ participation
moments and QIZ-on textile uptake/upgrading moments.

Relative to the old sequential routine, this script:
- fits one joint loss over QIZ-off and QIZ-on equilibria
- keeps the pre-QIZ participation targets on the off equilibrium
- targets uptake among Q,T US exporters on the on equilibrium
- allows a Q-specific US export-access shifter for textiles
- allows textile compliance-cost heterogeneity

The search strategy is deterministic coordinate descent on a cached objective.
That is deliberate: the model moments are lumpy on finite grids, so exact
gradient-based methods are not reliable here.
"""

from __future__ import annotations

import csv
import json
import math
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from calibrate_fixed_costs import (
    REGIONS,
    SECTORS,
    TARGETS,
    get_participation,
    get_qiz_firm_share,
    solve,
)
from qiz_model_ge import params_defensible


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "joint_calibration_summary.json")
OUT_CSV = os.path.join(ROOT, "joint_qiz_vs_noqiz_comparison.csv")
OUT_PARAMS = os.path.join(ROOT, "joint_calibrated_params.json")
CALIBRATION_SUMMARY = os.path.join(ROOT, "calibration_summary.json")
REVISED_BENCHMARK = os.path.join(ROOT, "revised_qiz_uptake_benchmark.json")
TEXTILE_QIZ_SHARE_MIN = 0.5 * TARGETS["qiz_firm_share"]["T"]

PARAM_ORDER = [
    "f_dom_T",
    "f_dom_O",
    "f_export_US_N_T",
    "q_us_access_ratio_T",
    "f_export_RW_T",
    "f_export_US_O",
    "f_export_RW_O",
    "f_entry_Q_T",
    "f_entry_Q_O",
    "fC_mean_T",
    "sigma_C_T",
    "f_upgrade_T",
]

BOUNDS = {
    "f_dom_T": (0.25, 3.00),
    "f_dom_O": (0.25, 3.00),
    "f_export_US_N_T": (1.00, 140.0),
    "q_us_access_ratio_T": (0.003, 1.00),
    "f_export_RW_T": (1.00, 160.0),
    "f_export_US_O": (0.20, 25.0),
    "f_export_RW_O": (0.20, 25.0),
    "f_entry_Q_T": (0.05, 2.50),
    "f_entry_Q_O": (0.05, 2.50),
    "fC_mean_T": (0.05, 8.00),
    "sigma_C_T": (0.05, 1.50),
    "f_upgrade_T": (0.20, 6.00),
}

PHASES = [
    {
        "label": "coarse",
        "single_factors": [0.70, 0.88, 1.00, 1.15, 1.35],
        "pair_factors": [0.82, 1.00, 1.22],
        "max_rounds": 1,
    },
    {
        "label": "medium",
        "single_factors": [0.85, 0.95, 1.00, 1.05, 1.15],
        "pair_factors": [0.90, 1.00, 1.10],
        "max_rounds": 1,
    },
    {
        "label": "fine",
        "single_factors": [0.95, 1.00, 1.05],
        "pair_factors": [0.96, 1.00, 1.04],
        "max_rounds": 1,
    },
]

PAIR_BLOCKS = [
    ("f_export_US_N_T", "q_us_access_ratio_T"),
    ("f_dom_T", "f_export_RW_T"),
    ("f_entry_Q_T", "q_us_access_ratio_T"),
    ("fC_mean_T", "sigma_C_T"),
]

SPECIAL_GRIDS = {
    "coarse": {
        "q_us_access_ratio_T": [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.40, 0.70, 1.00],
        "f_entry_Q_T": [0.15, 0.25, 0.35, 0.50, 0.70, 0.90, 1.20, 1.60],
        "fC_mean_T": [0.30, 0.60, 1.00, 1.50, 2.50, 4.00, 6.00],
        "sigma_C_T": [0.15, 0.30, 0.45, 0.60, 0.80, 1.00],
    },
    "medium": {
        "q_us_access_ratio_T": [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.06, 0.08, 0.12, 0.20],
        "f_entry_Q_T": [0.30, 0.40, 0.50, 0.70, 0.90, 1.10, 1.30],
        "fC_mean_T": [0.60, 1.00, 1.50, 2.00, 2.50, 3.50, 5.00],
        "sigma_C_T": [0.25, 0.40, 0.55, 0.70, 0.85],
    },
    "fine": {
        "q_us_access_ratio_T": [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06],
        "f_entry_Q_T": [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        "fC_mean_T": [1.00, 1.50, 2.00, 2.50, 3.00, 3.50],
        "sigma_C_T": [0.35, 0.45, 0.55, 0.65, 0.75],
    },
}


def clamp(name: str, value: float) -> float:
    lo, hi = BOUNDS[name]
    return min(max(float(value), lo), hi)


def load_starting_state() -> Dict[str, float]:
    state = {
        "f_dom_T": 1.01219,
        "f_dom_O": 1.00386,
        "f_export_US_N_T": 43.98199,
        "q_us_access_ratio_T": 0.02,
        "f_export_RW_T": 91.96667,
        "f_export_US_O": 3.51888,
        "f_export_RW_O": 3.58752,
        "f_entry_Q_T": 0.90,
        "f_entry_Q_O": 0.30,
        "fC_mean_T": 2.50,
        "sigma_C_T": 0.60,
        "f_upgrade_T": 3.00,
    }

    if os.path.exists(CALIBRATION_SUMMARY):
        with open(CALIBRATION_SUMMARY) as f:
            d = json.load(f)
        state["f_dom_T"] = float(d["f_dom"]["T"])
        state["f_dom_O"] = float(d["f_dom"]["O"])
        state["f_export_US_N_T"] = float(d["f_export_US"]["T"])
        state["f_export_RW_T"] = float(d["f_export_RW"]["T"])
        state["f_export_US_O"] = float(d["f_export_US"]["O"])
        state["f_export_RW_O"] = float(d["f_export_RW"]["O"])
        state["f_entry_Q_O"] = float(d["f_entry_Q"]["O"])

    if os.path.exists(REVISED_BENCHMARK):
        with open(REVISED_BENCHMARK) as f:
            d = json.load(f)
        state["f_entry_Q_T"] = max(state["f_entry_Q_T"], float(d["params"]["f_entry_Q"]["T"]))
        state["f_entry_Q_O"] = float(d["params"]["f_entry_Q"]["O"])
        state["fC_mean_T"] = float(d["params"]["fC_mean_T"])
        state["sigma_C_T"] = float(d["params"]["sigma_C_T"])
        state["f_upgrade_T"] = max(state["f_upgrade_T"], float(d["params"]["f_upgrade_T"]))

    return {k: clamp(k, v) for k, v in state.items()}


def load_revised_starting_state() -> Dict[str, float] | None:
    if not os.path.exists(REVISED_BENCHMARK):
        return None

    with open(REVISED_BENCHMARK) as f:
        d = json.load(f)

    params = d["params"]
    state = load_starting_state()
    n_us = float(params["f_export_US_N"]["T"])
    q_us = float(params["f_export_US_Q"]["T"])

    state["f_dom_T"] = float(params["f_dom"]["T"])
    state["f_dom_O"] = float(params["f_dom"]["O"])
    state["f_export_US_N_T"] = n_us
    state["q_us_access_ratio_T"] = q_us / max(n_us, 1.0e-12)
    state["f_export_RW_T"] = float(params["f_export_RW"]["T"])
    state["f_export_US_O"] = float(params["f_export_US_Q"]["O"])
    state["f_export_RW_O"] = float(params["f_export_RW"]["O"])
    state["f_entry_Q_T"] = float(params["f_entry_Q"]["T"])
    state["f_entry_Q_O"] = float(params["f_entry_Q"]["O"])
    state["fC_mean_T"] = float(params["fC_mean_T"])
    state["sigma_C_T"] = float(params["sigma_C_T"])
    state["f_upgrade_T"] = float(params["f_upgrade_T"])
    return {k: clamp(k, v) for k, v in state.items()}


def build_starting_states() -> List[Dict[str, float]]:
    seeds: List[Dict[str, float]] = [load_starting_state()]
    revised = load_revised_starting_state()
    if revised is not None:
        seeds.append(revised)

    anchor_sources = list(seeds)
    for base in anchor_sources:
        hi_share = deepcopy(base)
        hi_share["q_us_access_ratio_T"] = clamp("q_us_access_ratio_T", min(base["q_us_access_ratio_T"], 0.01))
        hi_share["f_entry_Q_T"] = clamp("f_entry_Q_T", min(base["f_entry_Q_T"], 0.35))
        seeds.append(hi_share)

    unique: List[Dict[str, float]] = []
    seen = set()
    for state in seeds:
        key = round_key(state)
        if key in seen:
            continue
        seen.add(key)
        unique.append(state)
    return unique


def build_params(state: Dict[str, float], n_phi: int = 25) -> Dict[str, Any]:
    p = params_defensible()
    p["roo_cost_formula"] = "normalized"
    p["n_phi"] = n_phi
    p["n_eps"] = 5
    p["sigma_C"]["T"] = state["sigma_C_T"]

    for r in REGIONS:
        p["f_dom"][(r, "T")] = state["f_dom_T"]
        p["f_dom"][(r, "O")] = state["f_dom_O"]
        p["f_export"][(r, "RW", "T")] = state["f_export_RW_T"]
        p["f_export"][(r, "US", "O")] = state["f_export_US_O"]
        p["f_export"][(r, "RW", "O")] = state["f_export_RW_O"]

    p["f_export"][("N", "US", "T")] = state["f_export_US_N_T"]
    p["f_export"][("Q", "US", "T")] = state["f_export_US_N_T"] * state["q_us_access_ratio_T"]

    p["f_entry"][("Q", "T")] = state["f_entry_Q_T"]
    p["f_entry"][("Q", "O")] = state["f_entry_Q_O"]
    p["f_entry"][("N", "T")] = 1.0
    p["f_entry"][("N", "O")] = 1.0

    p["fC_mean"]["T"] = state["fC_mean_T"]
    p["f_upgrade"]["T"] = state["f_upgrade_T"]
    return p


def round_key(state: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(round(float(state[k]), 8) for k in PARAM_ORDER)


def exact_penalty(actual: float, target: float, scale: float, weight: float) -> float:
    return weight * (((actual - target) / scale) ** 2)


def exporter_penalty(
    actual: float,
    target: float,
    scale: float,
    weight: float,
    above_weight: float = 0.30,
) -> float:
    diff = actual - target
    if diff > 0:
        diff *= above_weight
    return weight * ((diff / scale) ** 2)


def floor_penalty(actual: float, floor: float, scale: float, weight: float) -> float:
    shortfall = max(0.0, floor - actual)
    return weight * ((shortfall / scale) ** 2)


def extract_moments(off: Dict[str, Any], on: Dict[str, Any]) -> Dict[str, float]:
    part_t = get_participation(off, "T")
    part_o = get_participation(off, "O")
    q_t_off = off["moments"][("Q", "T")]
    q_t_on = on["moments"][("Q", "T")]
    return {
        "dom_only_T": float(part_t["dom_only"]),
        "dom_only_O": float(part_o["dom_only"]),
        "exp_US_T": float(part_t["exp_US"]),
        "exp_US_O": float(part_o["exp_US"]),
        "exp_RW_T": float(part_t["exp_RW"]),
        "exp_RW_O": float(part_o["exp_RW"]),
        "qiz_firm_share_T": float(get_qiz_firm_share(off, "T")),
        "qiz_firm_share_O": float(get_qiz_firm_share(off, "O")),
        "uptake_T": float(q_t_on["compliance_share_among_US_exporters"]),
        "upgrade_T": float(q_t_on["upgrade_share_among_active"]),
        "qt_us_share_on": float(q_t_on["US_export_share_among_active"]),
        "qt_active_on": float(q_t_on["active_share"]),
        "qt_active_off": float(q_t_off["active_share"]),
        "welfare_on": float(on["welfare"]),
        "welfare_off": float(off["welfare"]),
        "welfare_pct": float(100.0 * (on["welfare"] / off["welfare"] - 1.0)),
        "wQ_on": float(on["w"]["Q"]),
        "wQ_off": float(off["w"]["Q"]),
        "wN_on": float(on["w"]["N"]),
        "wN_off": float(off["w"]["N"]),
    }


def loss_breakdown(m: Dict[str, float]) -> Dict[str, float]:
    loss = {
        "dom_only_T": exact_penalty(m["dom_only_T"], TARGETS["dom_only"]["T"], scale=0.015, weight=1.2),
        "dom_only_O": exact_penalty(m["dom_only_O"], TARGETS["dom_only"]["O"], scale=0.015, weight=1.0),
        "exp_US_T": exporter_penalty(m["exp_US_T"], TARGETS["exp_US"]["T"], scale=0.010, weight=1.2),
        "exp_US_O": exporter_penalty(m["exp_US_O"], TARGETS["exp_US"]["O"], scale=0.006, weight=1.0),
        "exp_RW_T": exporter_penalty(m["exp_RW_T"], TARGETS["exp_RW"]["T"], scale=0.015, weight=1.0),
        "exp_RW_O": exporter_penalty(m["exp_RW_O"], TARGETS["exp_RW"]["O"], scale=0.020, weight=1.0),
        # Treat textile QIZ-region concentration as a soft target. The exact 0.889 target
        # is informative, but forcing it too hard shuts down the uptake margin.
        "qiz_firm_share_T": exact_penalty(
            m["qiz_firm_share_T"], TARGETS["qiz_firm_share"]["T"], scale=0.18, weight=0.35
        ),
        "qiz_firm_share_T_floor": floor_penalty(
            m["qiz_firm_share_T"], TEXTILE_QIZ_SHARE_MIN, scale=0.08, weight=2.0
        ),
        "qiz_firm_share_O": exact_penalty(
            m["qiz_firm_share_O"], TARGETS["qiz_firm_share"]["O"], scale=0.05, weight=0.9
        ),
        "uptake_T": exact_penalty(m["uptake_T"], TARGETS["compliance_rate"], scale=0.035, weight=1.6),
        "upgrade_T": exact_penalty(m["upgrade_T"], TARGETS["upgrading_rate"], scale=0.06, weight=1.0),
        "qt_us_share_on_floor": floor_penalty(m["qt_us_share_on"], floor=0.08, scale=0.04, weight=0.6),
        "qt_active_on_floor": floor_penalty(m["qt_active_on"], floor=0.08, scale=0.03, weight=0.5),
    }
    return loss


def total_loss(m: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    breakdown = loss_breakdown(m)
    return sum(breakdown.values()), breakdown


def record_priority(rec: Dict[str, Any]) -> Tuple[int, float, float, float]:
    if not rec.get("ok", False):
        return (2, float("inf"), float("inf"), float("inf"))

    qiz_share_t = float(rec["moments"]["qiz_firm_share_T"])
    shortfall = max(0.0, TEXTILE_QIZ_SHARE_MIN - qiz_share_t)
    feasible = 0 if shortfall <= 1.0e-12 else 1
    return (feasible, shortfall, float(rec["loss"]), -qiz_share_t)


def better_record(candidate: Dict[str, Any], incumbent: Dict[str, Any]) -> bool:
    return record_priority(candidate) < record_priority(incumbent)


class JointCalibrator:
    def __init__(self):
        self.cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}

    def evaluate(self, state: Dict[str, float], n_phi: int = 25) -> Dict[str, Any]:
        key = round_key(state) + (float(n_phi),)
        if key in self.cache:
            return self.cache[key]

        p = build_params(state, n_phi=n_phi)
        try:
            off = solve(p, qiz_on=False)
            on = solve(p, qiz_on=True, warm=off)
            moments = extract_moments(off, on)
            loss, breakdown = total_loss(moments)
            record = {
                "ok": True,
                "loss": float(loss),
                "loss_breakdown": breakdown,
                "moments": moments,
                "off": off,
                "on": on,
                "params": deepcopy(state),
            }
        except Exception as exc:
            try:
                on = solve(p, qiz_on=True)
                off = solve(p, qiz_on=False, warm=on)
                moments = extract_moments(off, on)
                loss, breakdown = total_loss(moments)
                record = {
                    "ok": True,
                    "loss": float(loss),
                    "loss_breakdown": breakdown,
                    "moments": moments,
                    "off": off,
                    "on": on,
                    "params": deepcopy(state),
                }
            except Exception:
                record = {
                    "ok": False,
                    "loss": float("inf"),
                    "loss_breakdown": {"solve_failure": 1.0e12},
                    "error": repr(exc),
                    "params": deepcopy(state),
                }

        self.cache[key] = record
        return record

    def search_single(self, base_state: Dict[str, float], name: str, phase: str, factors: List[float]) -> Dict[str, Any]:
        best = self.evaluate(base_state)
        special = SPECIAL_GRIDS.get(phase, {}).get(name)
        if special is not None:
            candidates = [clamp(name, value) for value in special]
        else:
            current = base_state[name]
            candidates = [clamp(name, current * factor) for factor in factors]
        for value in candidates:
            candidate = deepcopy(base_state)
            candidate[name] = value
            rec = self.evaluate(candidate)
            if better_record(rec, best):
                best = rec
        return best

    def search_pair(self, base_state: Dict[str, float], pair: Tuple[str, str], phase: str, factors: List[float]) -> Dict[str, Any]:
        best = self.evaluate(base_state)
        n1, n2 = pair
        v1, v2 = base_state[n1], base_state[n2]
        g1 = SPECIAL_GRIDS.get(phase, {}).get(n1)
        g2 = SPECIAL_GRIDS.get(phase, {}).get(n2)
        vals1 = [clamp(n1, x) for x in g1] if g1 is not None else [clamp(n1, v1 * f1) for f1 in factors]
        vals2 = [clamp(n2, x) for x in g2] if g2 is not None else [clamp(n2, v2 * f2) for f2 in factors]
        for x1 in vals1:
            for x2 in vals2:
                candidate = deepcopy(base_state)
                candidate[n1] = x1
                candidate[n2] = x2
                rec = self.evaluate(candidate)
                if better_record(rec, best):
                    best = rec
        return best

    def run(self, start_state: Dict[str, float]) -> Dict[str, Any]:
        best = self.evaluate(start_state)
        start_qiz_share = best.get("moments", {}).get("qiz_firm_share_T", float("nan"))
        print(
            f"start loss={best['loss']:.4f} qiz_share_T={start_qiz_share:.6f}",
            flush=True,
        )

        for phase in PHASES:
            print(f"\n== phase: {phase['label']} ==", flush=True)
            for rnd in range(phase["max_rounds"]):
                improved = False
                for name in PARAM_ORDER:
                    cand = self.search_single(best["params"], name, phase["label"], phase["single_factors"])
                    if better_record(cand, best):
                        best = cand
                        improved = True
                        print(
                            f"  single {name}: loss={best['loss']:.4f} "
                            f"qiz_share_T={best['moments']['qiz_firm_share_T']:.6f} "
                            f"value={best['params'][name]:.6f}",
                            flush=True,
                        )

                for pair in PAIR_BLOCKS:
                    cand = self.search_pair(best["params"], pair, phase["label"], phase["pair_factors"])
                    if better_record(cand, best):
                        best = cand
                        improved = True
                        v1 = best["params"][pair[0]]
                        v2 = best["params"][pair[1]]
                        print(
                            f"  pair {pair[0]}/{pair[1]}: loss={best['loss']:.4f} "
                            f"qiz_share_T={best['moments']['qiz_firm_share_T']:.6f} "
                            f"values=({v1:.6f}, {v2:.6f})",
                            flush=True,
                        )

                if not improved:
                    print("  no further improvement in this round", flush=True)
                    break

        return best


def floatify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: floatify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [floatify(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def write_csv(result: Dict[str, Any]) -> None:
    on = result["on"]
    off = result["off"]
    rows = [
        ["metric", "qiz_on", "qiz_off"],
        ["welfare", on["welfare"], off["welfare"]],
        ["welfare_pct_on_vs_off", 100.0 * (on["welfare"] / off["welfare"] - 1.0), ""],
        ["w_Q", on["w"]["Q"], off["w"]["Q"]],
        ["w_N", on["w"]["N"], off["w"]["N"]],
    ]
    for r in REGIONS:
        for s in SECTORS:
            on_m = on["moments"][(r, s)]
            off_m = off["moments"][(r, s)]
            prefix = f"{r}{s}"
            for key in [
                "active_share",
                "any_export_share_among_active",
                "domestic_only_share_among_active",
                "US_export_share_among_active",
                "RW_export_share_among_active",
                "compliance_share_among_active",
                "compliance_share_among_US_exporters",
                "upgrade_share_among_active",
            ]:
                rows.append([f"{prefix}_{key}", on_m.get(key, 0.0), off_m.get(key, 0.0)])

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def build_summary(calib: Dict[str, Any], final_eval: Dict[str, Any]) -> Dict[str, Any]:
    p = build_params(calib["params"], n_phi=80)
    summary = {
        "model_spec": {
            "roo_cost_formula": p["roo_cost_formula"],
            "n_phi_calibration": 15,
            "n_phi_final": 60,
            "n_eps": p["n_eps"],
            "uptake_denominator": "Q,T compliers divided by Q,T US exporters",
            "textile_q_us_access_shifter": "f_export[(Q,US,T)] / f_export[(N,US,T)]",
            "textile_qiz_share_minimum": TEXTILE_QIZ_SHARE_MIN,
        },
        "targets": floatify(TARGETS),
        "joint_loss": float(calib["loss"]),
        "loss_breakdown": floatify(calib["loss_breakdown"]),
        "params": floatify(calib["params"]),
        "moments_fit": floatify(final_eval["moments"]),
        "qiz_vs_noqiz": {
            "welfare_on": float(final_eval["on"]["welfare"]),
            "welfare_off": float(final_eval["off"]["welfare"]),
            "welfare_pct": float(100.0 * (final_eval["on"]["welfare"] / final_eval["off"]["welfare"] - 1.0)),
            "wQ_on": float(final_eval["on"]["w"]["Q"]),
            "wQ_off": float(final_eval["off"]["w"]["Q"]),
            "wN_on": float(final_eval["on"]["w"]["N"]),
            "wN_off": float(final_eval["off"]["w"]["N"]),
            "QT_on": floatify(final_eval["on"]["moments"][("Q", "T")]),
            "QT_off": floatify(final_eval["off"]["moments"][("Q", "T")]),
            "QO_on": floatify(final_eval["on"]["moments"][("Q", "O")]),
            "QO_off": floatify(final_eval["off"]["moments"][("Q", "O")]),
            "NT_on": floatify(final_eval["on"]["moments"][("N", "T")]),
            "NT_off": floatify(final_eval["off"]["moments"][("N", "T")]),
            "NO_on": floatify(final_eval["on"]["moments"][("N", "O")]),
            "NO_off": floatify(final_eval["off"]["moments"][("N", "O")]),
        },
    }
    return summary


def main() -> None:
    calibrator = JointCalibrator()
    best: Dict[str, Any] | None = None

    for idx, start in enumerate(build_starting_states(), start=1):
        print(
            f"\nStarting search {idx}: "
            f"q_us_access_ratio_T={start['q_us_access_ratio_T']:.6f}, "
            f"f_entry_Q_T={start['f_entry_Q_T']:.6f}",
            flush=True,
        )
        cand = calibrator.run(start)
        if best is None or better_record(cand, best):
            best = cand

    if best is None:
        raise RuntimeError("Joint calibration did not produce any candidate.")

    print("\nRe-evaluating winner on finer productivity grid...")
    final_eval = calibrator.evaluate(best["params"], n_phi=60)
    if not final_eval["ok"]:
        raise RuntimeError(f"Final evaluation failed: {final_eval.get('error')}")

    summary = build_summary(best, final_eval)

    with open(OUT_JSON, "w") as f:
        json.dump(floatify(summary), f, indent=2)
    with open(OUT_PARAMS, "w") as f:
        json.dump(floatify(best["params"]), f, indent=2)
    write_csv(final_eval)

    print("\nBest parameters:")
    for name in PARAM_ORDER:
        print(f"  {name} = {best['params'][name]:.6f}")

    print("\nFinal moments:")
    for key, value in final_eval["moments"].items():
        if isinstance(value, float) and math.isfinite(value):
            print(f"  {key} = {value:.6f}")

    print(f"\nSaved summary: {OUT_JSON}")
    print(f"Saved params:  {OUT_PARAMS}")
    print(f"Saved table:   {OUT_CSV}")


if __name__ == "__main__":
    main()
