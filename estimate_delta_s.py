#!/usr/bin/env python3
"""
estimate_delta_s.py
-------------------
Estimates the productivity upgrading multiplier delta_T from the
pre/post aggregated event study coefficients on non-US exports.

Identification: QIZ affects US market access directly via tariff
elimination. Any post-treatment increase in non-US exports must come
through productivity upgrading (not tariffs, not Israeli input costs
which affect all markets uniformly). This identifies delta_T via:

    delta_T = exp(beta_post_nonUS / (sigma_T - 1))

Source: prepost_aggregated.csv produced by paperv5 code.do Section 12.

Output: updates params_estimated.json with delta_s entry.
"""

import json
import math
import os
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

BASE     = os.path.dirname(__file__)
DATA     = os.path.join(BASE, "..", "Export and Import Data Egypt",
                         "Paper Figures v5", "prepost_aggregated.csv")
OUT      = os.path.join(BASE, "params_estimated.json")

# ── parameters ────────────────────────────────────────────────────────────────

SIGMA_T  = 6.7   # within-sector CES elasticity for textiles

# ── load event study results ──────────────────────────────────────────────────

df = pd.read_csv(DATA).set_index("outcome")

beta_post = df.loc["NonUS_log", "post_b"]
se_post   = df.loc["NonUS_log", "post_se"]

# ── compute delta_T ───────────────────────────────────────────────────────────

log_delta  = beta_post / (SIGMA_T - 1)
delta_T    = math.exp(log_delta)

# 95% CI by error propagation
log_delta_lo = (beta_post - 1.96 * se_post) / (SIGMA_T - 1)
log_delta_hi = (beta_post + 1.96 * se_post) / (SIGMA_T - 1)
ci_lo = math.exp(log_delta_lo)
ci_hi = math.exp(log_delta_hi)

print("=" * 55)
print("  Productivity Upgrading Multiplier delta_T")
print("=" * 55)
print(f"  beta_post (NonUS log): {beta_post:.4f}  (SE={se_post:.4f})")
print(f"  sigma_T - 1:           {SIGMA_T - 1:.1f}")
print(f"  log(delta_T):          {log_delta:.4f}")
print(f"  delta_T:               {delta_T:.4f}")
print(f"  95% CI:                [{ci_lo:.4f}, {ci_hi:.4f}]")
print("=" * 55)

# ── save ─────────────────────────────────────────────────────────────────────

if os.path.exists(OUT):
    with open(OUT, "r") as f:
        params = json.load(f)
else:
    params = {}

params["delta_s"] = {
    "description": "Productivity upgrading multiplier: phi' = delta_s * phi for upgrading firms",
    "grouping_1": {"T": round(delta_T, 4), "O": 1.0},
    "grouping_2": {"T": round(delta_T, 4), "S1": 1.0, "S2": 1.0, "S3": 1.0, "O": 1.0},
    "details": {
        "T": {
            "delta":     round(delta_T, 4),
            "ci_low":    round(ci_lo, 4),
            "ci_high":   round(ci_hi, 4),
            "post_coef": beta_post,
            "post_se":   se_post,
            "sigma_used": SIGMA_T,
            "formula":   f"delta = exp(post_coef / (sigma - 1)) = exp({beta_post:.4f} / {SIGMA_T-1:.1f})"
        },
        "O": {"delta": 1.0, "note": "QIZ applies to textiles only; no upgrading for other sectors"}
    },
    "source": "Matched firm panel (final_data_matched.dta), CEM-weighted event study on log non-US exports",
    "source_file": "Export and Import Data Egypt/Paper Figures v5/prepost_aggregated.csv",
    "method": "Post-period average coefficient from paperv5 code.do Section 12, inverted via delta = exp(beta_post / (sigma_T - 1))",
    "script": "paperv5 code.do Section 12 -> prepost_aggregated.csv -> estimate_delta_s.py"
}

with open(OUT, "w") as f:
    json.dump(params, f, indent=2)

print(f"\n  Saved to: {OUT}")
