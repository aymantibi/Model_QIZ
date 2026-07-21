"""Targeted refinement for the reformed QIZ calibration.

This keeps the preferred economic specification:

- upgrading is available without QIZ;
- QIZ compliance adds an upgrading/productivity complementarity;
- compliance is heterogeneous;
- some firms have an MFN US-incumbency advantage.

The grid is intentionally local around the first reformed fit so it can be run
reliably with the current GE solver.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import calibrate_reformed_qiz as cr
from calibrate_clean_stacked_qiz import clean_float


OUT_JSON = Path("refined_reformed_qiz_calibration.json")
OUT_CSV = Path("refined_reformed_qiz_grid.csv")
OUT_MD = Path("refined_reformed_qiz_results.md")

TARGET_COMP = 0.3207126948775056
TARGET_COMP_RW = 109.76195087187696


def refined_loss(row: Dict[str, Any]) -> float:
    rw = row["comp_rw_pct"]
    terms = []
    terms.append(((row["T_comp_US"] - TARGET_COMP) / 0.055) ** 2)
    terms.append(((rw - TARGET_COMP_RW) / 55.0) ** 2 if rw is not None else 100.0)
    terms.append((row["O_comp_US"] / 0.04) ** 2)
    terms.append(0.5 * cr.prepolicy_loss(row["prepolicy_fit"]))

    # Keep the intended mechanism in the selected point.
    terms.append((max(0.0, 0.05 - row["T_upgrade_intensity_change"]) / 0.06) ** 2)
    terms.append((max(0.0, 0.75 - row["comp_avg_upgrade_intensity"]) / 0.50) ** 2)
    terms.append((max(0.0, 0.25 - row["upgrade_intensity_gap_comp_noncomp"]) / 0.50) ** 2)
    return float(sum(terms))


def run_refinement() -> Dict[str, Any]:
    targets = cr.compute_empirical_targets()
    base = cr.build_params(n_phi=70, n_eps=7, intensity_grid=5)

    grid = {
        "mfn_incumbency_T": [1.8, 2.0, 2.2],
        "sigma_C_T": [1.2],
        "fC_T": [25.0, 40.0, 60.0, 80.0, 120.0],
        "upgrade_psi_T": [0.03],
        "upgrade_psi_comp_T": [0.04, 0.05, 0.06, 0.08, 0.10],
        "upgrade_cost_quad_T": [3.0, 4.0, 5.0],
        "upgrade_cost_comp_mult_T": [1.0],
    }

    rows: List[Dict[str, Any]] = []
    off_cache: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    off_seed: Dict[str, Any] | None = None

    for mfn_incumbency_T in grid["mfn_incumbency_T"]:
        for upgrade_psi_T in grid["upgrade_psi_T"]:
            for upgrade_cost_quad_T in grid["upgrade_cost_quad_T"]:
                off_key = (mfn_incumbency_T, upgrade_psi_T, upgrade_cost_quad_T)
                p_off = cr.set_policy_params(
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
                    off_cache[off_key] = cr.solve_off(p_off, seed=off_seed)
                    off_seed = off_cache[off_key]
                except RuntimeError as exc:
                    rows.append(
                        {
                            "case": "refined_reformed_qiz",
                            "mfn_incumbency_T": mfn_incumbency_T,
                            "upgrade_psi_T": upgrade_psi_T,
                            "upgrade_cost_quad_T": upgrade_cost_quad_T,
                            "error": f"off failed: {exc}",
                            "loss": float("inf"),
                        }
                    )
                    continue

                for sigma_C_T in grid["sigma_C_T"]:
                    for fC_T in grid["fC_T"]:
                        for upgrade_psi_comp_T in grid["upgrade_psi_comp_T"]:
                            for upgrade_cost_comp_mult_T in grid["upgrade_cost_comp_mult_T"]:
                                p = cr.set_policy_params(
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
                                    ev = cr.evaluate(p, fixed_off=off_cache[off_key])
                                except RuntimeError as exc:
                                    rows.append(
                                        {
                                            "case": "refined_reformed_qiz",
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
                                    continue

                                row = {
                                    "case": "refined_reformed_qiz",
                                    "mfn_incumbency_T": mfn_incumbency_T,
                                    "sigma_C_T": sigma_C_T,
                                    "fC_T": fC_T,
                                    "upgrade_psi_T": upgrade_psi_T,
                                    "upgrade_psi_comp_T": upgrade_psi_comp_T,
                                    "upgrade_cost_quad_T": upgrade_cost_quad_T,
                                    "upgrade_cost_comp_mult_T": upgrade_cost_comp_mult_T,
                                    **ev,
                                }
                                row["loss"] = refined_loss(row)
                                rows.append(row)

    valid = [r for r in rows if "error" not in r and math.isfinite(float(r["loss"]))]
    best = min(valid, key=lambda r: r["loss"]) if valid else None
    closest_compliance = min(valid, key=lambda r: abs(r["T_comp_US"] - TARGET_COMP)) if valid else None
    best_nonus_near_comp = None
    near = [r for r in valid if abs(r["T_comp_US"] - TARGET_COMP) <= 0.08 and r["comp_rw_pct"] is not None]
    if near:
        best_nonus_near_comp = max(near, key=lambda r: r["comp_rw_pct"])

    return {
        "targets": targets,
        "grid": grid,
        "numeric_grid": {"n_phi": base["n_phi"], "n_eps": base["n_eps"], "intensity_grid": base["upgrade_intensity_grid_size"]},
        "rows": rows,
        "best": best,
        "closest_compliance": closest_compliance,
        "best_nonus_near_compliance": best_nonus_near_comp,
    }


def write_csv(rows: Iterable[Dict[str, Any]]) -> None:
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
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(summary: Dict[str, Any]) -> None:
    best = summary["best"]
    closest = summary["closest_compliance"]
    best_nonus = summary["best_nonus_near_compliance"]

    def fmt(x: Any, digits: int = 3) -> str:
        return "N/A" if x is None else f"{float(x):.{digits}f}"

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# Refined Reformed QIZ Calibration\n\n")
        f.write("The refinement keeps the QIZ-upgrading channel and searches locally around the first reformed fit.\n\n")
        f.write("## Targets\n\n")
        f.write(f"- Textile compliance among US exporters: `{TARGET_COMP:.3f}`\n")
        f.write(f"- Complier non-US export growth: `{TARGET_COMP_RW:.1f}%`\n\n")
        if best is None:
            f.write("No valid refinement point found.\n")
            return

        f.write("## Best Point\n\n")
        for key in [
            "mfn_incumbency_T",
            "sigma_C_T",
            "fC_T",
            "upgrade_psi_T",
            "upgrade_psi_comp_T",
            "upgrade_cost_quad_T",
            "upgrade_cost_comp_mult_T",
        ]:
            f.write(f"- `{key} = {best[key]}`\n")
        f.write(f"- Textile compliance among US exporters: `{best['T_comp_US']:.3f}`\n")
        f.write(f"- Textile QIZ compliers among active firms: `{best['T_comp_active']:.3f}`\n")
        f.write(f"- Textile MFN US exporters among active firms: `{best['T_MFN_US_active']:.3f}`\n")
        f.write(f"- Textile US exporters among active firms: `{best['T_US_active']:.3f}`\n")
        f.write(
            f"- Upgrade intensity, no-QIZ to QIZ: "
            f"`{best['T_upgrade_intensity_off_active']:.3f}` to "
            f"`{best['T_upgrade_intensity_on_active']:.3f}`\n"
        )
        f.write(f"- Complier upgrade intensity: `{best['comp_avg_upgrade_intensity']:.3f}`\n")
        f.write(f"- Noncomplier upgrade intensity: `{best['noncomp_avg_upgrade_intensity']:.3f}`\n")
        f.write(f"- Complier non-US export change: `{fmt(best['comp_rw_pct'], 1)}%`\n")
        f.write(f"- Textile-sector non-US export change: `{best['T_sector_RW_pct']:.1f}%`\n")
        f.write(f"- Welfare gain: `{best['welfare_pct']:.3f}%`\n\n")

        f.write("## Nearby Diagnostics\n\n")
        if closest is not None:
            f.write(
                f"- Closest compliance: `T_comp_US = {closest['T_comp_US']:.3f}`, "
                f"`comp_rw_pct = {fmt(closest['comp_rw_pct'], 1)}%`, "
                f"`welfare = {closest['welfare_pct']:.3f}%`\n"
            )
        if best_nonus is not None:
            f.write(
                f"- Highest non-US response near compliance: "
                f"`T_comp_US = {best_nonus['T_comp_US']:.3f}`, "
                f"`comp_rw_pct = {fmt(best_nonus['comp_rw_pct'], 1)}%`, "
                f"`welfare = {best_nonus['welfare_pct']:.3f}%`\n"
            )
        f.write("\n## Interpretation\n\n")
        f.write(
            "The refined local grid preserves the key fix: compliance is interior and "
            "upgrading is possible without QIZ but stronger among QIZ compliers. "
            "The remaining limitation is that matching the full non-US export growth "
            "target still requires a fairly strong QIZ-upgrading channel.\n"
        )


def main() -> None:
    summary = run_refinement()
    OUT_JSON.write_text(json.dumps(clean_float(summary), indent=2), encoding="utf-8")
    write_csv(summary["rows"])
    write_md(clean_float(summary))
    best = summary["best"]
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    if best:
        print(
            "Best:",
            f"mfn={best['mfn_incumbency_T']}",
            f"fC={best['fC_T']}",
            f"psi_comp={best['upgrade_psi_comp_T']}",
            f"qcost={best['upgrade_cost_quad_T']}",
            f"T_comp_US={best['T_comp_US']:.3f}",
            f"comp_rw={best['comp_rw_pct']:.1f}",
            f"W={best['welfare_pct']:.3f}",
        )


if __name__ == "__main__":
    main()
