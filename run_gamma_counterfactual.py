#!/usr/bin/env python3
"""
Run ROO-content counterfactuals and plot equilibrium outcomes against gamma.

Default behavior:
- Uses preset: fit_exports_dual_market
- Varies textiles ROO requirement gamma_T from 0% to 30% in 5pp steps
- Solves full GE at each point (warm-started)
- Writes:
    1) CSV with equilibrium outcomes by gamma
    2) PNG multi-panel figure
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qiz_model_ge as m


def run_path(
    p: Dict[str, Any],
    sector: str,
    gamma_grid_pct: List[float],
    disable_upgrade: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    state = None

    for g_pct in gamma_grid_pct:
        g = float(g_pct) / 100.0
        gamma_override = dict(p["gamma"])
        gamma_override[sector] = g

        sol = m.solve_equilibrium(
            p,
            qiz_on=True,
            gamma_override=gamma_override,
            disable_upgrade=disable_upgrade,
            initial_state=state,
            verbose=False,
        )
        state = sol

        tr = m.summarize_trade(sol, p)
        rw_q = sol["w"]["Q"] / max(sol["goods"]["P_EG"], 1e-12)
        rw_n = sol["w"]["N"] / max(sol["goods"]["P_EG"], 1e-12)

        rows.append(
            {
                "gamma_pct": g_pct,
                "welfare": sol["welfare"],
                "wQ": sol["w"]["Q"],
                "wN": sol["w"]["N"],
                "real_wage_Q": rw_q,
                "real_wage_N": rw_n,
                "LQ_share": sol["Ls"]["Q"] / p["L_total"],
                "comp_share_QT": sol["moments"][("Q", "T")]["compliance_share_among_active"],
                "US_export_share_QT": sol["moments"][("Q", "T")]["US_export_share_among_active"],
                "RW_export_share_QT": sol["moments"][("Q", "T")]["RW_export_share_among_active"],
                "upgrade_share_QT": sol["moments"][("Q", "T")]["upgrade_share_among_active"],
                "upgrade_intensity_QT": sol["moments"][("Q", "T")]["upgrade_intensity_among_active"],
                "X_Q_US": tr["exports"][("Q", "US")],
                "X_Q_RW": tr["exports"][("Q", "RW")],
                "X_total_US": tr["exports"][("Q", "US")] + tr["exports"][("N", "US")],
                "X_total_RW": tr["exports"][("Q", "RW")] + tr["exports"][("N", "RW")],
            }
        )

    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, png_path: Path, title: str) -> None:
    metrics = [
        ("welfare", "Welfare (Y/P)"),
        ("real_wage_Q", "Real Wage: Q"),
        ("real_wage_N", "Real Wage: N"),
        ("LQ_share", "Employment Share in Q"),
        ("comp_share_QT", "Compliance Share (Q,T)"),
        ("upgrade_intensity_QT", "Upgrade Intensity (Q,T)"),
        ("US_export_share_QT", "US Export Share (Q,T active)"),
        ("X_Q_US", "Q Exports to US"),
        ("X_Q_RW", "Q Exports to Non-US"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)

    x = df["gamma_pct"].to_numpy()
    for ax, (col, label) in zip(axes.flat, metrics):
        y = df[col].to_numpy()
        ax.plot(x, y, color="#1f77b4", linewidth=2.0, marker="o", markersize=5)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Israeli Input Requirement gamma (%)")
        ax.set_ylabel(label)
        ax.set_xticks(x)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROO gamma counterfactual path and plot outcomes.")
    parser.add_argument(
        "--preset",
        default="fit_exports_dual_market",
        choices=[
            "paper",
            "institutional_transparent",
            "interior_gamma",
            "data_like",
            "upgrade_complementarity",
            "fit_exports_dual_market",
            "fit_us_aggregate_positive",
        ],
    )
    parser.add_argument("--sector", default="T", choices=["T", "O"])
    parser.add_argument("--gamma-start", type=float, default=0.0)
    parser.add_argument("--gamma-end", type=float, default=30.0)
    parser.add_argument("--gamma-step", type=float, default=5.0)
    parser.add_argument("--disable-upgrade", action="store_true")
    parser.add_argument("--out-csv", default="gamma_counterfactual_results.csv")
    parser.add_argument("--out-png", default="gamma_counterfactual_figure.png")
    args = parser.parse_args()

    p = m.build_params_from_preset(args.preset)
    gamma_grid_pct = list(np.arange(args.gamma_start, args.gamma_end + 1e-9, args.gamma_step))

    df = run_path(
        p=p,
        sector=args.sector,
        gamma_grid_pct=gamma_grid_pct,
        disable_upgrade=args.disable_upgrade,
    )

    csv_path = Path(args.out_csv)
    png_path = Path(args.out_png)
    df.to_csv(csv_path, index=False)

    title = f"ROO Counterfactual Path | preset={args.preset} | sector={args.sector}"
    if args.disable_upgrade:
        title += " | upgrading=OFF"
    make_figure(df, png_path, title)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {png_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
