#!/usr/bin/env python3
"""
Generate presentation-ready QIZ model outputs.

This script does three things:
1. Uses a stable local calibration that prioritizes the complier moments.
2. Compares model-implied QIZ vs No-QIZ outcomes to the user's target facts.
3. Runs the ROO-input counterfactual and tracks Israeli input use.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qiz_model_ge as m


TARGET_MOMENTS: List[Tuple[str, str, float, bool]] = [
    ("micro_comp_us_pct", "Complier exports to US", 150.0, True),
    ("micro_comp_rw_pct", "Complier exports to non-US", 150.0, True),
    ("target_real_wage_q_pct", "Real wage in treated region Q", 15.0, False),
    ("agg_total_us_pct", "Aggregate exports to US", 100.0, False),
    ("agg_q_t_us_pct", "Textile exports to US", 400.0, False),
]


def build_presentation_params() -> Dict[str, Any]:
    """
    Stable local calibration chosen to move the model closer to the complier moments.

    This is intentionally transparent: it starts from the existing
    `fit_us_aggregate_positive` preset and applies a few local changes.
    The key one for the gamma mechanism is restoring non-textile ROO costs,
    so lower gamma can pull in low-tariff sectors as new compliers.
    """
    p = m.build_params_from_preset("fit_us_aggregate_positive")
    p["upgrade_psi"]["T"] = 0.08
    p["t_mfn"]["T"] = 0.10
    p["gamma"]["O"] = 0.105
    p["fC_mean"]["O"] = 0.40
    p["p_il"]["O"] = 1.10
    p["xi_admin"]["O"] = 0.60
    return p


def pct_change(on_v: float, off_v: float) -> float:
    if abs(off_v) < 1.0e-12:
        return float("nan")
    return 100.0 * (on_v - off_v) / abs(off_v)


def make_target_table(comp: Dict[str, Any], p: Dict[str, Any]) -> pd.DataFrame:
    moms = m.collect_stylized_moments(comp, p)
    rows: List[Dict[str, Any]] = []
    for key, label, target, targeted in TARGET_MOMENTS:
        model_v = float(moms.get(key, np.nan))
        rows.append(
            {
                "moment_key": key,
                "label": label,
                "target_pct": target,
                "model_pct": model_v,
                "gap_pct_points": model_v - target,
                "targeted_in_fit": targeted,
            }
        )
    return pd.DataFrame(rows)


def make_validation_table(comp: Dict[str, Any], p: Dict[str, Any]) -> pd.DataFrame:
    on = comp["on"]
    off = comp["off"]
    tr_on = m.summarize_trade(on, p)
    tr_off = m.summarize_trade(off, p)
    sec_on = m.summarize_trade_by_sector(on, p)
    sec_off = m.summarize_trade_by_sector(off, p)

    rows: List[Dict[str, Any]] = []

    def add_row(group: str, destination: str, on_v: float, off_v: float) -> None:
        rows.append(
            {
                "group": group,
                "destination": destination,
                "qiz_on": float(on_v),
                "no_qiz": float(off_v),
                "pct_change": float(pct_change(on_v, off_v)),
            }
        )

    for s, label in [("T", "Textiles"), ("O", "Other manufacturing")]:
        for j in ["US", "RW"]:
            add_row(
                label,
                j,
                sum(sec_on[(r, s, j)] for r in p["regions"]),
                sum(sec_off[(r, s, j)] for r in p["regions"]),
            )

    for r, label in [("Q", "QIZ regions"), ("N", "Non-QIZ regions")]:
        for j in ["US", "RW"]:
            add_row(label, j, tr_on["exports"][(r, j)], tr_off["exports"][(r, j)])

    for j in ["US", "RW"]:
        add_row(
            "All sectors",
            j,
            sum(tr_on["exports"][(r, j)] for r in p["regions"]),
            sum(tr_off["exports"][(r, j)] for r in p["regions"]),
        )

    return pd.DataFrame(rows)


def summarize_israeli_input_use(
    sol: Dict[str, Any],
    p: Dict[str, Any],
) -> Dict[str, float]:
    """
    Model-implied Israeli intermediate spending embodied in compliant US shipments.

    This is not a customs-data import series. It is the equilibrium spending on the
    Israeli-content share of the intermediate bundle for compliant exporters.
    """
    out: Dict[str, float] = {"total": 0.0}
    gamma_map = sol.get("gamma", p["gamma"])

    for s in p["sectors"]:
        out[f"sector_{s}"] = 0.0

    if not sol.get("qiz_on", True):
        return out

    eps_grid, w_eps = m.normal_grid(p["n_eps"], a=p["eps_std"])

    for s in p["sectors"]:
        gamma_s = float(gamma_map[s])
        if gamma_s <= 0.0:
            continue

        phi_grid, w_phi = m.pareto_grid(p["phi_min"][s], p["theta"][s], p["n_phi"])
        r = "Q"
        sector_total = 0.0

        for phi, w_phi_i in zip(phi_grid, w_phi):
            for eps, w_eps_i in zip(eps_grid, w_eps):
                wt = w_phi_i * w_eps_i
                best = m.firm_best(
                    phi,
                    eps,
                    r,
                    s,
                    sol["w"][r],
                    sol["goods"]["P_EG_s"][s],
                    sol["goods"]["E_EG_s"][s],
                    p,
                    gamma_override=gamma_map,
                    qiz_on=sol.get("qiz_on", True),
                    disable_upgrade=sol.get("disable_upgrade", False),
                )
                if not (best["compliance"] and best["serve"]["US"]):
                    continue

                us_variable_cost = ((p["sigma"][s] - 1.0) / p["sigma"][s]) * best["R"]["US"]
                il_spending = gamma_s * (1.0 - p["alpha"][s]) * us_variable_cost
                sector_total += sol["M"][(r, s)] * wt * il_spending

        out[f"sector_{s}"] = float(sector_total)
        out["total"] += float(sector_total)

    return out


def _safe_avg(total: float, mass: float) -> float:
    if mass <= 1.0e-12:
        return float("nan")
    return float(total / mass)


def summarize_gamma_mechanism(
    baseline_sol: Dict[str, Any],
    current_sol: Dict[str, Any],
    p: Dict[str, Any],
    baseline_gamma: Dict[str, float],
    current_gamma: Dict[str, float],
) -> pd.DataFrame:
    """
    Decompose lower-gamma effects into incumbent-complier intensification
    and newly induced compliers by sector within the QIZ region.
    """
    rows: List[Dict[str, Any]] = []
    eps_grid, w_eps = m.normal_grid(p["n_eps"], a=p["eps_std"])
    r = "Q"

    for s in p["sectors"]:
        phi_grid, w_phi = m.pareto_grid(p["phi_min"][s], p["theta"][s], p["n_phi"])
        active_mass_current = 0.0
        incumbent_type_mass = 0.0
        new_type_mass = 0.0
        lost_type_mass = 0.0
        current_comp_type_mass = 0.0

        inc_us_base = inc_us_cur = 0.0
        inc_rw_base = inc_rw_cur = 0.0
        inc_u_base = inc_u_cur = 0.0
        new_us_cur = new_rw_cur = new_u_cur = 0.0

        for phi, w_phi_i in zip(phi_grid, w_phi):
            for eps, w_eps_i in zip(eps_grid, w_eps):
                wt = w_phi_i * w_eps_i
                base = m.firm_best(
                    phi,
                    eps,
                    r,
                    s,
                    baseline_sol["w"][r],
                    baseline_sol["goods"]["P_EG_s"][s],
                    baseline_sol["goods"]["E_EG_s"][s],
                    p,
                    gamma_override=baseline_gamma,
                    qiz_on=True,
                    disable_upgrade=baseline_sol.get("disable_upgrade", False),
                )
                cur = m.firm_best(
                    phi,
                    eps,
                    r,
                    s,
                    current_sol["w"][r],
                    current_sol["goods"]["P_EG_s"][s],
                    current_sol["goods"]["E_EG_s"][s],
                    p,
                    gamma_override=current_gamma,
                    qiz_on=True,
                    disable_upgrade=current_sol.get("disable_upgrade", False),
                )

                if any(cur["serve"].values()):
                    active_mass_current += wt
                if cur["compliance"]:
                    current_comp_type_mass += wt

                if base["compliance"] and cur["compliance"]:
                    incumbent_type_mass += wt
                    inc_us_base += wt * base["R"]["US"]
                    inc_us_cur += wt * cur["R"]["US"]
                    inc_rw_base += wt * base["R"]["RW"]
                    inc_rw_cur += wt * cur["R"]["RW"]
                    inc_u_base += wt * base.get("upgrade_intensity", 0.0)
                    inc_u_cur += wt * cur.get("upgrade_intensity", 0.0)
                elif (not base["compliance"]) and cur["compliance"]:
                    new_type_mass += wt
                    new_us_cur += wt * cur["R"]["US"]
                    new_rw_cur += wt * cur["R"]["RW"]
                    new_u_cur += wt * cur.get("upgrade_intensity", 0.0)
                elif base["compliance"] and (not cur["compliance"]):
                    lost_type_mass += wt

        current_mass_multiplier = current_sol["M"][(r, s)]
        rows.append(
            {
                "sector": s,
                "incumbent_complier_mass_current": float(current_mass_multiplier * incumbent_type_mass),
                "new_complier_mass_current": float(current_mass_multiplier * new_type_mass),
                "lost_complier_type_mass": float(lost_type_mass),
                "current_complier_mass": float(current_mass_multiplier * current_comp_type_mass),
                "active_mass_current": float(current_mass_multiplier * active_mass_current),
                "new_complier_share_of_current_compliers": float(
                    new_type_mass / max(current_comp_type_mass, 1.0e-12)
                ),
                "new_complier_share_of_active_current": float(
                    new_type_mass / max(active_mass_current, 1.0e-12)
                ),
                "incumbent_us_pct_vs_baseline": float(
                    pct_change(_safe_avg(inc_us_cur, incumbent_type_mass), _safe_avg(inc_us_base, incumbent_type_mass))
                ),
                "incumbent_rw_pct_vs_baseline": float(
                    pct_change(_safe_avg(inc_rw_cur, incumbent_type_mass), _safe_avg(inc_rw_base, incumbent_type_mass))
                ),
                "incumbent_upgrade_intensity_change": float(
                    _safe_avg(inc_u_cur, incumbent_type_mass) - _safe_avg(inc_u_base, incumbent_type_mass)
                ),
                "new_complier_avg_us_exports": float(_safe_avg(new_us_cur, new_type_mass)),
                "new_complier_avg_rw_exports": float(_safe_avg(new_rw_cur, new_type_mass)),
                "new_complier_avg_upgrade_intensity": float(_safe_avg(new_u_cur, new_type_mass)),
            }
        )

    return pd.DataFrame(rows)


def run_gamma_counterfactual(
    p: Dict[str, Any],
    gamma_grid_pct: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    mechanism_rows: List[Dict[str, Any]] = []
    baseline_gamma = {s: p["gamma"][s] for s in p["sectors"]}
    baseline_pct = 100.0 * baseline_gamma[p["sectors"][0]]
    baseline_sol = m.solve_equilibrium(
        p,
        qiz_on=True,
        gamma_override=baseline_gamma,
        disable_upgrade=False,
        verbose=False,
    )
    state = baseline_sol

    for gamma_pct in gamma_grid_pct:
        # ROO is treated as economy-wide in this presentation exercise:
        # the same Israeli-input requirement applies to all sectors.
        gamma_override = {s: gamma_pct / 100.0 for s in p["sectors"]}

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

        total_us = sum(tr["exports"][(r, "US")] for r in p["regions"])
        total_rw = sum(tr["exports"][(r, "RW")] for r in p["regions"])
        textile_us = sum(sec[(r, "T", "US")] for r in p["regions"])
        textile_rw = sum(sec[(r, "T", "RW")] for r in p["regions"])
        mechanism_df = summarize_gamma_mechanism(
            baseline_sol=baseline_sol,
            current_sol=sol,
            p=p,
            baseline_gamma=baseline_gamma,
            current_gamma=gamma_override,
        )
        for row in mechanism_df.to_dict(orient="records"):
            row["gamma_pct"] = float(gamma_pct)
            row["baseline_gamma_pct"] = float(baseline_pct)
            mechanism_rows.append(row)

        rows.append(
            {
                "gamma_pct": float(gamma_pct),
                "baseline_gamma_pct": float(baseline_pct),
                "welfare": float(sol["welfare"]),
                "real_wage_Q": float(sol["w"]["Q"] / sol["goods"]["P_EG"]),
                "real_wage_N": float(sol["w"]["N"] / sol["goods"]["P_EG"]),
                "comp_share_QT": float(sol["moments"][("Q", "T")]["compliance_share_among_active"]),
                "comp_share_QO": float(sol["moments"][("Q", "O")]["compliance_share_among_active"]),
                "upgrade_intensity_QT": float(sol["moments"][("Q", "T")]["upgrade_intensity_among_active"]),
                "upgrade_intensity_QO": float(sol["moments"][("Q", "O")]["upgrade_intensity_among_active"]),
                "exports_total_US": float(total_us),
                "exports_total_RW": float(total_rw),
                "exports_textiles_US": float(textile_us),
                "exports_textiles_RW": float(textile_rw),
                "israeli_inputs_total": float(il["total"]),
                "israeli_inputs_textiles": float(il["sector_T"]),
                "israeli_inputs_other": float(il["sector_O"]),
                "israeli_inputs_per_us_export": float(il["total"] / max(total_us, 1.0e-12)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(mechanism_rows)


def plot_target_fit(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)

    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["target_pct"], width, label="Target", color="#c4a35a")
    ax.bar(x + width / 2, df["model_pct"], width, label="Model", color="#1f6e73")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Percent change")
    ax.set_title("Targeted Moments vs Model")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=18, ha="right")
    ax.legend()

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    for ax, dest, title in zip(axes, ["US", "RW"], ["Exports to US", "Exports to non-US"]):
        sub = df[df["destination"] == dest].copy()
        x = np.arange(len(sub))
        width = 0.38
        ax.bar(x - width / 2, sub["no_qiz"], width, label="No QIZ", color="#8da0cb")
        ax.bar(x + width / 2, sub["qiz_on"], width, label="QIZ", color="#fc8d62")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"], rotation=22, ha="right")
        ax.set_ylabel("Level")
        if dest == "US":
            ax.legend()

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)

    x = df["gamma_pct"].to_numpy()
    baseline_gamma = float(df["baseline_gamma_pct"].iloc[0])

    axes[0, 0].plot(x, df["comp_share_QT"], marker="o", color="#1f6e73", linewidth=2.0, label="Textiles")
    axes[0, 0].plot(x, df["comp_share_QO"], marker="s", color="#b35b2d", linewidth=2.0, label="Other manufacturing")
    axes[0, 0].set_title("Compliance Share in QIZ Region")
    axes[0, 0].set_xlabel("Israeli input requirement (%)")
    axes[0, 0].set_ylabel("Share")
    axes[0, 0].axvline(baseline_gamma, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].legend()

    axes[0, 1].plot(x, df["exports_total_US"], marker="o", color="#b35b2d", linewidth=2.0, label="Total exports to US")
    axes[0, 1].plot(x, df["exports_textiles_US"], marker="s", color="#c19446", linewidth=2.0, label="Textile exports to US")
    axes[0, 1].set_title("US Export Response")
    axes[0, 1].set_xlabel("Israeli input requirement (%)")
    axes[0, 1].set_ylabel("Exports")
    axes[0, 1].axvline(baseline_gamma, color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].legend()

    axes[1, 0].plot(x, df["real_wage_Q"], marker="o", color="#4c78a8", linewidth=2.0, label="Real wage Q")
    axes[1, 0].plot(x, df["welfare"], marker="s", color="#59a14f", linewidth=2.0, label="Welfare")
    axes[1, 0].set_title("Wages and Welfare")
    axes[1, 0].set_xlabel("Israeli input requirement (%)")
    axes[1, 0].axvline(baseline_gamma, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].legend()

    axes[1, 1].plot(x, df["israeli_inputs_total"], marker="o", color="#7a3b69", linewidth=2.0, label="Israeli inputs")
    axes[1, 1].plot(
        x,
        df["israeli_inputs_per_us_export"],
        marker="s",
        color="#2f4b7c",
        linewidth=2.0,
        label="Israeli inputs / US exports",
    )
    axes[1, 1].set_title("Input-Use Tradeoff")
    axes[1, 1].set_xlabel("Israeli input requirement (%)")
    axes[1, 1].axvline(baseline_gamma, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].legend()

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_mechanism(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)

    baseline_gamma = float(df["baseline_gamma_pct"].iloc[0])
    specs = [
        ("incumbent_us_pct_vs_baseline", "Incumbent Compliers: US Exports", "% vs baseline gamma"),
        ("incumbent_upgrade_intensity_change", "Incumbent Compliers: Upgrading", "Change in intensity"),
        ("new_complier_mass_current", "New Compliers Induced by Lower Gamma", "Mass"),
        ("new_complier_share_of_current_compliers", "Share of Current Compliers That Are New", "Share"),
    ]

    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        for sector, label, color, marker in [
            ("T", "Textiles", "#1f6e73", "o"),
            ("O", "Other manufacturing", "#b35b2d", "s"),
        ]:
            sub = df[df["sector"] == sector].sort_values("gamma_pct")
            ax.plot(sub["gamma_pct"], sub[metric], marker=marker, color=color, linewidth=2.0, label=label)
        ax.axvline(baseline_gamma, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Economy-wide Israeli input requirement (%)")
        ax.set_ylabel(ylabel)

    axes[0, 0].legend()
    axes[0, 1].legend()

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run presentation-ready QIZ analysis.")
    parser.add_argument("--outdir", default="presentation_outputs")
    parser.add_argument("--gamma-start", type=float, default=0.0)
    parser.add_argument("--gamma-end", type=float, default=30.0)
    parser.add_argument("--gamma-step", type=float, default=5.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = build_presentation_params()
    comp = m.compare_qiz_on_off(p, disable_upgrade=False)

    target_df = make_target_table(comp, p)
    validation_df = make_validation_table(comp, p)

    gamma_grid_pct = list(np.arange(args.gamma_start, args.gamma_end + 1.0e-9, args.gamma_step))
    gamma_df, mechanism_df = run_gamma_counterfactual(p, gamma_grid_pct)

    target_csv = outdir / "presentation_target_fit.csv"
    validation_csv = outdir / "presentation_validation_exports.csv"
    gamma_csv = outdir / "presentation_gamma_tradeoff.csv"
    mechanism_csv = outdir / "presentation_gamma_mechanism.csv"
    target_png = outdir / "presentation_target_fit.png"
    validation_png = outdir / "presentation_validation_exports.png"
    gamma_png = outdir / "presentation_gamma_tradeoff.png"
    mechanism_png = outdir / "presentation_gamma_mechanism.png"
    summary_json = outdir / "presentation_summary.json"

    target_df.to_csv(target_csv, index=False)
    validation_df.to_csv(validation_csv, index=False)
    gamma_df.to_csv(gamma_csv, index=False)
    mechanism_df.to_csv(mechanism_csv, index=False)

    plot_target_fit(target_df, target_png)
    plot_validation(validation_df, validation_png)
    plot_gamma_tradeoff(gamma_df, gamma_png)
    plot_gamma_mechanism(mechanism_df, mechanism_png)

    summary = {
        "calibration": {
            "base_preset": "fit_us_aggregate_positive",
            "local_changes": {
                "baseline_gamma_pct": p["gamma"]["T"] * 100.0,
                "upgrade_psi_T": p["upgrade_psi"]["T"],
                "t_mfn_T": p["t_mfn"]["T"],
                "gamma_O": p["gamma"]["O"],
                "fC_mean_O": p["fC_mean"]["O"],
                "p_il_O": p["p_il"]["O"],
                "xi_admin_O": p["xi_admin"]["O"],
            },
        },
        "target_fit_table": target_df.to_dict(orient="records"),
        "gamma_grid_pct": gamma_grid_pct,
        "files": {
            "target_csv": str(target_csv),
            "validation_csv": str(validation_csv),
            "gamma_csv": str(gamma_csv),
            "mechanism_csv": str(mechanism_csv),
            "target_png": str(target_png),
            "validation_png": str(validation_png),
            "gamma_png": str(gamma_png),
            "mechanism_png": str(mechanism_png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {target_csv}")
    print(f"Saved: {validation_csv}")
    print(f"Saved: {gamma_csv}")
    print(f"Saved: {mechanism_csv}")
    print(f"Saved: {target_png}")
    print(f"Saved: {validation_png}")
    print(f"Saved: {gamma_png}")
    print(f"Saved: {mechanism_png}")
    print(f"Saved: {summary_json}")
    print()
    print("Target moments")
    print(target_df.to_string(index=False))
    print()
    print("Validation exports")
    print(validation_df.to_string(index=False))
    print()
    print("Gamma tradeoff")
    print(gamma_df.to_string(index=False))
    print()
    print("Gamma mechanism")
    print(mechanism_df.to_string(index=False))


if __name__ == "__main__":
    main()
