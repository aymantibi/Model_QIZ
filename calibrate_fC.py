#!/usr/bin/env python3
"""
calibrate_fC.py
---------------
Calibrates the compliance fixed cost fC_mean["T"] to match the observed
compliance rate among US-exporting textile firms.

Target moment:
    compliance_share_among_active (Q, T) = 0.35
    Source: estimate_compliance_rate.do, pooled 2006-2008 average.

Method:
    Binary search over fC_mean["T"]. For each candidate value, solve the
    full equilibrium with QIZ on and read off the predicted compliance rate.
    Find the value that matches the target.

Usage:
    python calibrate_fC.py
"""

import json
import os
import numpy as np
from scipy.optimize import brentq
from qiz_model_ge import params_defensible, solve_equilibrium

# ── target ───────────────────────────────────────────────────────────────────

TARGET_COMP_RATE = 0.35   # pooled 2006-2008 compliance rate
SECTOR          = "T"
REGION          = "Q"
OUT_PATH        = os.path.join(os.path.dirname(__file__), "params_estimated.json")

# ── helper ────────────────────────────────────────────────────────────────────

def predicted_compliance_rate(fC: float, p: dict) -> float:
    """Solve equilibrium and return compliance_share_among_active for (Q, T)."""
    p_try = {k: v for k, v in p.items()}
    p_try["fC_mean"] = {**p["fC_mean"], SECTOR: fC}
    sol = solve_equilibrium(p_try, qiz_on=True, verbose=False)
    return sol["moments"][(REGION, SECTOR)]["compliance_share_among_active"]


def objective(fC: float, p: dict) -> float:
    pred = predicted_compliance_rate(fC, p)
    print(f"    fC = {fC:.4f}  ->  predicted rate = {pred:.4f}  (target = {TARGET_COMP_RATE:.4f})")
    return pred - TARGET_COMP_RATE


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = params_defensible()

    # Update with estimated parameters
    p["sigma"]    = {"T": 6.7,    "O": 6.0}
    p["theta"]    = {"T": 6.7152, "O": 6.0}
    p["alpha"]    = {"T": 0.1855, "O": 0.0641}
    p["beta"]     = {"T": 0.0521, "O": 0.9479}
    p["t_mfn"]    = {"T": 0.0960, "O": 0.0095}
    p["gamma"]    = {"T": 0.105,  "O": 0.105}
    p["delta"]    = {"T": 1.178,  "O": 1.0}

    print("=" * 60)
    print("  Calibrating fC_mean[T] to match compliance rate")
    print(f"  Target: {TARGET_COMP_RATE:.4f}")
    print("=" * 60)

    # --- bracket: find fC_lo and fC_hi that straddle the target ---
    print("\nStep 1: Bracketing...")
    fC_lo, fC_hi = 0.01, 5.0

    rate_lo = predicted_compliance_rate(fC_lo, p)
    rate_hi = predicted_compliance_rate(fC_hi, p)

    print(f"  fC = {fC_lo:.3f}  ->  rate = {rate_lo:.4f}")
    print(f"  fC = {fC_hi:.3f}  ->  rate = {rate_hi:.4f}")

    if rate_lo < TARGET_COMP_RATE:
        print("  WARNING: even at fC=0.01 predicted rate is below target.")
        print("  Check model parameters — compliance may be too costly.")
    elif rate_hi > TARGET_COMP_RATE:
        print("  WARNING: even at fC=5.0 predicted rate is above target.")
        print("  Try a wider upper bound.")
    else:
        # --- brentq root-finding ---
        print("\nStep 2: Root-finding (Brent's method)...")
        fC_star = brentq(
            objective, fC_lo, fC_hi,
            args=(p,),
            xtol=1e-4, rtol=1e-4,
            maxiter=30
        )

        rate_star = predicted_compliance_rate(fC_star, p)

        print("\n" + "=" * 60)
        print(f"  RESULT: fC_mean[T] = {fC_star:.4f}")
        print(f"  Predicted compliance rate: {rate_star:.4f}")
        print(f"  Target compliance rate:    {TARGET_COMP_RATE:.4f}")
        print("=" * 60)

        # --- save to params_estimated.json ---
        if os.path.exists(OUT_PATH):
            with open(OUT_PATH, "r") as f:
                params = json.load(f)
        else:
            params = {}

        params["fC_s"] = {
            "description": "Compliance fixed cost in labor units (fC_mean[T] calibrated to match compliance rate)",
            "grouping_1": {
                "T": round(fC_star, 4),
                "O": round(p["fC_mean"]["O"], 4)
            },
            "target_moment": {
                "compliance_share_among_active": TARGET_COMP_RATE,
                "source": "estimate_compliance_rate.do, pooled 2006-2008"
            },
            "predicted_moment": round(rate_star, 4),
            "method": "Brent root-finding on solve_equilibrium() output",
            "note": "sigma_C = 0 (deterministic fixed cost, cutoff compliance rule)"
        }

        with open(OUT_PATH, "w") as f:
            json.dump(params, f, indent=2)

        print(f"\n  Saved to: {OUT_PATH}")
