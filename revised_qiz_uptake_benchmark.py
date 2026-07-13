#!/usr/bin/env python3
"""
Build a revised QIZ benchmark that generates positive textile uptake.

Key changes relative to the original baseline:
- normalized ROO cost formula
- textile compliance heterogeneity (n_eps > 1, sigma_C[T] > 0)
- lower Q-region US export fixed cost for textiles
- solve QIZ-off first, then warm-start QIZ-on from the off equilibrium
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Tuple

from calibrate_fixed_costs import REGIONS, SECTORS, fast_p, solve
from qiz_model_ge import params_defensible


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "revised_qiz_uptake_benchmark.json")
OUT_CSV = os.path.join(ROOT, "revised_qiz_uptake_comparison.csv")
BASELINE_SUMMARY = os.path.join(ROOT, "calibration_summary.json")


def load_stage1_costs() -> Dict[str, Dict[str, float]]:
    """
    Reuse the latest stage-1 fixed-cost estimates when available.
    Fall back to the most recent successful baseline run in this workspace.
    """
    if os.path.exists(BASELINE_SUMMARY):
        with open(BASELINE_SUMMARY) as f:
            d = json.load(f)
        return {
            "f_dom": {k: float(v) for k, v in d["f_dom"].items()},
            "f_export_US": {k: float(v) for k, v in d["f_export_US"].items()},
            "f_export_RW": {k: float(v) for k, v in d["f_export_RW"].items()},
        }

    return {
        "f_dom": {"T": 1.01219, "O": 1.0038606896551725},
        "f_export_US": {"T": 43.98199302673339, "O": 3.518880310058594},
        "f_export_RW": {"T": 91.96666950225827, "O": 3.587517395019532},
    }


def qiz_firm_share(sol: Dict[str, Any], sector: str) -> float:
    M = sol["M"]
    cache = sol["goods"]["cache"]
    n_q = M[("Q", sector)] * cache[("Q", sector)]["E_active"]
    n_n = M[("N", sector)] * cache[("N", sector)]["E_active"]
    return n_q / max(n_q + n_n, 1e-12)


def build_params() -> Dict[str, Any]:
    costs = load_stage1_costs()
    p = fast_p(params_defensible())

    # Use the uptake-friendly ROO specification and activate heterogeneity.
    p["roo_cost_formula"] = "normalized"
    p["n_phi"] = 15
    p["n_eps"] = 5
    p["sigma_C"]["T"] = 0.6

    for s in SECTORS:
        for r in REGIONS:
            p["f_dom"][(r, s)] = costs["f_dom"][s]
            p["f_export"][(r, "US", s)] = costs["f_export_US"][s]
            p["f_export"][(r, "RW", s)] = costs["f_export_RW"][s]

    # Softened entry calibration for textiles and existing O calibration.
    p["f_entry"][("Q", "T")] = 0.5
    p["f_entry"][("N", "T")] = 1.0
    p["f_entry"][("Q", "O")] = 0.3
    p["f_entry"][("N", "O")] = 1.0

    # Give Q-textile firms a lower fixed cost of serving the US market.
    p["f_export"][("Q", "US", "T")] = p["f_export"][("N", "US", "T")] * 0.02

    # Calibrate uptake on the interior branch.
    p["fC_mean"]["T"] = 2.5
    return p


def collect_summary(p: Dict[str, Any], off: Dict[str, Any], on: Dict[str, Any]) -> Dict[str, Any]:
    qt_on = on["moments"][("Q", "T")]
    qt_off = off["moments"][("Q", "T")]
    return {
        "params": {
            "roo_cost_formula": p["roo_cost_formula"],
            "n_phi": p["n_phi"],
            "n_eps": p["n_eps"],
            "sigma_C_T": p["sigma_C"]["T"],
            "f_dom": {"T": p["f_dom"][("Q", "T")], "O": p["f_dom"][("Q", "O")]},
            "f_export_US_Q": {"T": p["f_export"][("Q", "US", "T")], "O": p["f_export"][("Q", "US", "O")]},
            "f_export_US_N": {"T": p["f_export"][("N", "US", "T")], "O": p["f_export"][("N", "US", "O")]},
            "f_export_RW": {"T": p["f_export"][("Q", "RW", "T")], "O": p["f_export"][("Q", "RW", "O")]},
            "f_entry_Q": {"T": p["f_entry"][("Q", "T")], "O": p["f_entry"][("Q", "O")]},
            "f_entry_N": {"T": p["f_entry"][("N", "T")], "O": p["f_entry"][("N", "O")]},
            "fC_mean_T": p["fC_mean"]["T"],
            "f_upgrade_T": p["f_upgrade"]["T"],
        },
        "targets": {
            "uptake_among_QT_US_exporters": 0.321,
            "qiz_firm_share_T_soft_reference": 0.5233713305077827,
        },
        "moments": {
            "off_QT": qt_off,
            "on_QT": qt_on,
            "off_qiz_share_T": qiz_firm_share(off, "T"),
            "on_qiz_share_T": qiz_firm_share(on, "T"),
            "uptake_among_QT_US_exporters": qt_on["compliance_share_among_US_exporters"],
            "welfare_off": off["welfare"],
            "welfare_on": on["welfare"],
            "welfare_pct_on_vs_off": 100.0 * (on["welfare"] / off["welfare"] - 1.0),
            "wQ_off": off["w"]["Q"],
            "wQ_on": on["w"]["Q"],
            "wN_off": off["w"]["N"],
            "wN_on": on["w"]["N"],
        },
    }


def write_csv(summary: Dict[str, Any]) -> None:
    rows = [
        ["metric", "qiz_on", "qiz_off"],
        ["welfare", summary["moments"]["welfare_on"], summary["moments"]["welfare_off"]],
        ["welfare_pct_on_vs_off", summary["moments"]["welfare_pct_on_vs_off"], ""],
        ["w_Q", summary["moments"]["wQ_on"], summary["moments"]["wQ_off"]],
        ["w_N", summary["moments"]["wN_on"], summary["moments"]["wN_off"]],
    ]
    for key in [
        "active_share",
        "domestic_only_share_among_active",
        "US_export_share_among_active",
        "RW_export_share_among_active",
        "compliance_share_among_active",
        "compliance_share_among_US_exporters",
        "upgrade_share_among_active",
    ]:
        rows.append([f"QT_{key}", summary["moments"]["on_QT"][key], summary["moments"]["off_QT"][key]])

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    p = build_params()
    off = solve(p, qiz_on=False)
    on = solve(p, qiz_on=True, warm=off)

    summary = collect_summary(p, off, on)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(summary)

    print(json.dumps(summary, indent=2))
    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
