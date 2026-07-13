#!/usr/bin/env python3
"""
Plot targeted QIZ outcomes:
- Q textile exports to US and RW (QIZ on vs off)
- Same-type complier export changes to US and RW (percent)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qiz_model_ge as m


def collect_targeted_rows(p: Dict[str, Any], disable_upgrade: bool = False) -> pd.DataFrame:
    comp = m.compare_qiz_on_off(p, disable_upgrade=disable_upgrade)
    on = comp["on"]
    off = comp["off"]
    sec_on = m.summarize_trade_by_sector(on, p)
    sec_off = m.summarize_trade_by_sector(off, p)
    same = m.summarize_qt_same_type_changes(on, off, p, disable_upgrade=disable_upgrade)

    rows = []
    for key, label in [
        (("Q", "T", "US"), "Q Textile Exports to US"),
        (("Q", "T", "RW"), "Q Textile Exports to Non-US"),
    ]:
        on_v = sec_on[key]
        off_v = sec_off[key]
        pct = 100.0 * (on_v - off_v) / max(abs(off_v), 1e-12)
        rows.append(
            {
                "metric": label,
                "on_value": float(on_v),
                "off_value": float(off_v),
                "pct_change_on_vs_off": float(pct),
            }
        )

    rows.append(
        {
            "metric": "Complier Same-Type US Exports (% change)",
            "on_value": np.nan,
            "off_value": np.nan,
            "pct_change_on_vs_off": float(same["comp"]["US_pct"]) if np.isfinite(same["comp"]["US_pct"]) else np.nan,
        }
    )
    rows.append(
        {
            "metric": "Complier Same-Type Non-US Exports (% change)",
            "on_value": np.nan,
            "off_value": np.nan,
            "pct_change_on_vs_off": float(same["comp"]["RW_pct"]) if np.isfinite(same["comp"]["RW_pct"]) else np.nan,
        }
    )
    rows.append(
        {
            "metric": "Complier Share Among Active Q,T",
            "on_value": float(on["moments"][("Q", "T")]["compliance_share_among_active"]),
            "off_value": float(off["moments"][("Q", "T")]["compliance_share_among_active"]),
            "pct_change_on_vs_off": np.nan,
        }
    )

    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_png: Path, title: str) -> None:
    lvl = df[df["on_value"].notna()].copy()
    pct = df[df["pct_change_on_vs_off"].notna()].copy()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    x = np.arange(len(lvl))
    width = 0.36
    axes[0].bar(x - width / 2, lvl["off_value"], width, label="No QIZ", color="#8da0cb")
    axes[0].bar(x + width / 2, lvl["on_value"], width, label="QIZ", color="#fc8d62")
    axes[0].set_title("Targeted Levels")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(lvl["metric"], rotation=20, ha="right")
    axes[0].legend()

    x2 = np.arange(len(pct))
    axes[1].bar(x2, pct["pct_change_on_vs_off"], color="#66c2a5")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Targeted Percent Changes (QIZ vs No QIZ)")
    axes[1].set_ylabel("%")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(pct["metric"], rotation=20, ha="right")

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot targeted QIZ outcomes.")
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
    parser.add_argument("--out-csv", default="qiz_targeted_effects.csv")
    parser.add_argument("--out-png", default="qiz_targeted_effects.png")
    args = parser.parse_args()

    p = m.build_params_from_preset(args.preset)
    df = collect_targeted_rows(p, disable_upgrade=args.disable_upgrade)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    df.to_csv(out_csv, index=False)

    title = f"Targeted QIZ Outcomes | preset={args.preset}"
    if args.disable_upgrade:
        title += " | upgrading=OFF"
    make_plot(df, out_png, title)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved figure: {out_png}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
