#!/usr/bin/env python3
"""
Create a QIZ-on vs QIZ-off comparison table and figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qiz_model_ge as m


def build_rows(p: Dict[str, Any], disable_upgrade: bool = False) -> pd.DataFrame:
    comp = m.compare_qiz_on_off(p, disable_upgrade=disable_upgrade)
    on = comp["on"]
    off = comp["off"]
    tr_on = comp["trade_on"]
    tr_off = comp["trade_off"]

    def pct(a: float, b: float) -> float:
        if abs(b) < 1.0e-8:
            return float("nan")
        return 100.0 * (a - b) / abs(b)

    rows: List[Dict[str, float]] = []
    metrics = {
        "welfare": (on["welfare"], off["welfare"]),
        "real_wage_Q": (on["w"]["Q"] / on["goods"]["P_EG"], off["w"]["Q"] / off["goods"]["P_EG"]),
        "real_wage_N": (on["w"]["N"] / on["goods"]["P_EG"], off["w"]["N"] / off["goods"]["P_EG"]),
        "LQ_share": (on["Ls"]["Q"] / p["L_total"], off["Ls"]["Q"] / p["L_total"]),
        "X_Q_US": (tr_on["exports"][("Q", "US")], tr_off["exports"][("Q", "US")]),
        "X_Q_RW": (tr_on["exports"][("Q", "RW")], tr_off["exports"][("Q", "RW")]),
        "X_total_US": (
            tr_on["exports"][("Q", "US")] + tr_on["exports"][("N", "US")],
            tr_off["exports"][("Q", "US")] + tr_off["exports"][("N", "US")],
        ),
        "X_total_RW": (
            tr_on["exports"][("Q", "RW")] + tr_on["exports"][("N", "RW")],
            tr_off["exports"][("Q", "RW")] + tr_off["exports"][("N", "RW")],
        ),
        "comp_share_QT": (
            on["moments"][("Q", "T")]["compliance_share_among_active"],
            off["moments"][("Q", "T")]["compliance_share_among_active"],
        ),
    }

    for metric, (on_v, off_v) in metrics.items():
        rows.append(
            {
                "metric": metric,
                "on_value": float(on_v),
                "off_value": float(off_v),
                "pct_change_on_vs_off": float(pct(on_v, off_v)),
            }
        )

    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_png: Path, title: str) -> None:
    labels = df["metric"].tolist()
    on_v = df["on_value"].to_numpy()
    off_v = df["off_value"].to_numpy()
    pct_v = df["pct_change_on_vs_off"].to_numpy()

    x = np.arange(len(labels))
    width = 0.38

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)

    axes[0].bar(x - width / 2, off_v, width, label="No QIZ", color="#8da0cb")
    axes[0].bar(x + width / 2, on_v, width, label="QIZ", color="#fc8d62")
    axes[0].set_title("Levels by Outcome")
    axes[0].set_ylabel("Level")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].legend()

    axes[1].bar(x, pct_v, color="#66c2a5")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Percent Change: QIZ vs No QIZ")
    axes[1].set_ylabel("% change (on vs off)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="QIZ on vs off figure.")
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
    parser.add_argument("--disable-upgrade", action="store_true")
    parser.add_argument("--out-csv", default="qiz_vs_noqiz_comparison.csv")
    parser.add_argument("--out-png", default="qiz_vs_noqiz_comparison.png")
    args = parser.parse_args()

    p = m.build_params_from_preset(args.preset)
    df = build_rows(p, disable_upgrade=args.disable_upgrade)

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    df.to_csv(out_csv, index=False)

    title = f"QIZ vs No-QIZ | preset={args.preset}"
    if args.disable_upgrade:
        title += " | upgrading=OFF"
    make_plot(df, out_png, title)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved figure: {out_png}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
