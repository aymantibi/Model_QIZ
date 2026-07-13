#!/usr/bin/env python3
"""
Continuous-upgrade QIZ benchmark.

This keeps the joint-fit trade and entry structure from the latest runs, but
replaces the binary upgrade block with a continuous upgrade choice where QIZ
compliance raises the return to upgrading for textile firms.

The goal is to make upgrade a QIZ complement rather than a hard requirement.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict

from calibrate_fixed_costs import solve
from qiz_model_ge import params_institutional_transparent


ROOT = os.path.dirname(__file__)
OUT_JSON = os.path.join(ROOT, "upgrade_complementarity_benchmark.json")
OUT_CSV = os.path.join(ROOT, "upgrade_complementarity_qiz_vs_noqiz.csv")


def build_params() -> Dict[str, Any]:
    p = params_institutional_transparent()

    # Keep the latest stable trade/entry structure from the joint-fit candidate.
    for r in ["Q", "N"]:
        p["f_dom"][(r, "T")] = 1.01219
        p["f_dom"][(r, "O")] = 1.0038606896551725
        p["f_export"][(r, "RW", "T")] = 64.37666865158079
        p["f_export"][(r, "US", "O")] = 2.0
        p["f_export"][(r, "RW", "O")] = 3.587517395019532

    p["f_export"][("N", "US", "T")] = 28.0
    p["f_export"][("Q", "US", "T")] = 28.0 * 0.02

    p["f_entry"][("Q", "T")] = 1.2
    p["f_entry"][("N", "T")] = 1.0
    p["f_entry"][("Q", "O")] = 0.3
    p["f_entry"][("N", "O")] = 1.0

    # Compliance target is closest to the observed 0.321 around this region.
    p["n_eps"] = 5
    p["fC_mean"]["T"] = 14.0

    # Continuous upgrade with QIZ complementarity.
    p["upgrade_mode"] = "continuous"
    p["upgrade_requires_US"] = False
    p["upgrade_psi"]["T"] = 0.12
    p["upgrade_psi_comp"]["T"] = 0.095
    p["upgrade_cost_fixed"]["T"] = 0.0
    p["upgrade_cost_lin"]["T"] = 0.0
    p["upgrade_cost_quad"]["T"] = 4.5
    p["upgrade_intensity_grid_size"] = 7
    p["upgrade_intensity_max"]["T"] = 2.0

    return p


def solve_pair(p: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        off = solve(p, qiz_on=False)
        on = solve(p, qiz_on=True, warm=off)
        return off, on
    except Exception:
        on = solve(p, qiz_on=True)
        off = solve(p, qiz_on=False, warm=on)
        return off, on


def floatify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: floatify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [floatify(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def write_csv(off: Dict[str, Any], on: Dict[str, Any]) -> None:
    rows = [
        ["metric", "qiz_on", "qiz_off"],
        ["welfare", on["welfare"], off["welfare"]],
        ["welfare_pct_on_vs_off", 100.0 * (on["welfare"] / off["welfare"] - 1.0), ""],
        ["w_Q", on["w"]["Q"], off["w"]["Q"]],
        ["w_N", on["w"]["N"], off["w"]["N"]],
    ]

    for key, label in [(("Q", "T"), "QT"), (("N", "T"), "NT"), (("Q", "O"), "QO"), (("N", "O"), "NO")]:
        on_m = on["moments"][key]
        off_m = off["moments"][key]
        for moment in [
            "active_share",
            "any_export_share_among_active",
            "domestic_only_share_among_active",
            "US_export_share_among_active",
            "RW_export_share_among_active",
            "compliance_share_among_active",
            "compliance_share_among_US_exporters",
            "upgrade_share_among_active",
            "upgrade_intensity_among_active",
        ]:
            rows.append([f"{label}_{moment}", on_m.get(moment, 0.0), off_m.get(moment, 0.0)])

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    p = build_params()
    off, on = solve_pair(p)

    summary = {
        "spec": {
            "preset": "params_institutional_transparent + continuous upgrade complementarity",
            "upgrade_mode": p["upgrade_mode"],
            "upgrade_requires_US": p["upgrade_requires_US"],
            "n_phi": p["n_phi"],
            "n_eps": p["n_eps"],
        },
        "params": {
            "f_dom_T": p["f_dom"][("Q", "T")],
            "f_dom_O": p["f_dom"][("Q", "O")],
            "f_export_US_N_T": p["f_export"][("N", "US", "T")],
            "f_export_US_Q_T": p["f_export"][("Q", "US", "T")],
            "f_export_RW_T": p["f_export"][("Q", "RW", "T")],
            "f_export_US_O": p["f_export"][("Q", "US", "O")],
            "f_export_RW_O": p["f_export"][("Q", "RW", "O")],
            "f_entry_Q_T": p["f_entry"][("Q", "T")],
            "f_entry_Q_O": p["f_entry"][("Q", "O")],
            "fC_mean_T": p["fC_mean"]["T"],
            "sigma_C_T": p["sigma_C"]["T"],
            "upgrade_psi_T": p["upgrade_psi"]["T"],
            "upgrade_psi_comp_T": p["upgrade_psi_comp"]["T"],
            "upgrade_cost_fixed_T": p["upgrade_cost_fixed"]["T"],
            "upgrade_cost_quad_T": p["upgrade_cost_quad"]["T"],
        },
        "moments": {
            "QT_on": floatify(on["moments"][("Q", "T")]),
            "QT_off": floatify(off["moments"][("Q", "T")]),
            "NT_on": floatify(on["moments"][("N", "T")]),
            "NT_off": floatify(off["moments"][("N", "T")]),
            "QO_on": floatify(on["moments"][("Q", "O")]),
            "QO_off": floatify(off["moments"][("Q", "O")]),
            "NO_on": floatify(on["moments"][("N", "O")]),
            "NO_off": floatify(off["moments"][("N", "O")]),
            "welfare_on": on["welfare"],
            "welfare_off": off["welfare"],
            "welfare_pct_on_vs_off": 100.0 * (on["welfare"] / off["welfare"] - 1.0),
            "wQ_on": on["w"]["Q"],
            "wQ_off": off["w"]["Q"],
            "wN_on": on["w"]["N"],
            "wN_off": off["w"]["N"],
        },
        "fit_notes": {
            "uptake_target_reference": 0.321,
            "upgrade_target_reference": 0.491,
            "comment": (
                "This benchmark is designed to make upgrading more attractive for "
                "QIZ-compliant textile firms. In this parameter region, QIZ mainly "
                "raises upgrade intensity; the share of upgraders moves only when the "
                "complementarity is pushed much harder, which then overshoots uptake."
            ),
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(floatify(summary), f, indent=2)
    write_csv(off, on)

    print(json.dumps(floatify(summary), indent=2))
    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
