#!/usr/bin/env python3
"""
QIZ Heterogeneous-Firm Trade Model (2 regions, multi-sector, US-only ROO, endogenous compliance + upgrading)

- Regions: r in {Q, N}
- Sectors: s in {T, O}  (Textiles/Apparel, Other manufacturing)  <-- extendable
- Destinations: j in {EG, US, RW}

Key mechanisms:
1) Endogenous export participation by destination with fixed costs.
2) Endogenous ROO compliance (Q-only), which affects only US-destined production costs and removes MFN tariff.
   ROO composite is *normalized* Cobb–Douglas so it does not mechanically inflate costs.
   A simple admin wedge chi_s(gamma) = 1 + xi_s * gamma increases with stringency.
   Compliance fixed cost is heterogeneous (lognormal) to generate partial take-up.
3) Export-triggered productivity upgrading: upgrade is available only if the firm serves the US.
   Upgrading scales productivity by delta_s and costs w_r fU_s.

General equilibrium inside Egypt:
- Nested CES across sectors (eta) and within sectors (sigma_s).
- Endogenous domestic sector price indices P_EG,s and aggregate P_EG.
- Logit labor mobility across regions based on real wages (kappa).
- Free entry in each (r,s): expected operating profit equals entry cost w_r fE_rs.
- Regional labor market clearing: labor supply equals variable + fixed + entry labor demand.

Counterfactuals:
A) Shut down the productivity channel (set delta_s=1 or set fU_s very high) and compare outcomes.
B) Vary ROO content requirement gamma_s (e.g., textiles) and compute welfare frontier W(gamma).

Run:
  python qiz_model_ge.py

You can edit parameters in params_defensible().
"""

from __future__ import annotations
import argparse
import copy
import itertools
import json
from datetime import datetime, timezone
import numpy as np
from typing import Dict, Tuple, List, Any


# -----------------------------
# Parameters / calibration
# -----------------------------

def _load_absorption_2004() -> dict:
    """Load E_js from absorption_2004.json (grouping_1: T and O only)."""
    import os, json
    path = os.path.join(os.path.dirname(__file__), "data_calibration", "absorption_2004.json")
    with open(path) as f:
        ab = json.load(f)["grouping_1"]
    return ab


def _load_trade_costs_2004() -> dict:
    """
    Load EK-inverted composite trade wedges tau_ijs from trade_costs.json (grouping_1).

    tau_ijs = (pi_jjs / pi_ijs)^{1/theta_s} is the composite wedge for exporter i
    selling to destination j in sector s.  It embeds both iceberg trade costs and
    relative TFP/wage differences and is used directly as d_iceberg[(r,j,s)] in the
    model.  Both Q and N regions are assigned the same Egypt-aggregate tau since
    we do not have separate QIZ/non-QIZ export data to identify region-level wedges.
    """
    import os, json
    path = os.path.join(os.path.dirname(__file__), "trade_costs.json")
    with open(path) as f:
        d = json.load(f)["grouping_1"]["data"]
    # tau_EGY_j  = composite wedge for Egypt exporting to destination j
    # notation in file: tau_{exporter}_{destination}
    out = {}
    for s in ["T", "O"]:
        tc = d[s]["trade_costs"]
        out[s] = {
            "EG":  tc["tau_EGY_EGY"],   # = 1.0 by construction
            "US":  tc["tau_EGY_US"],
            "RW":  tc["tau_EGY_RoW"],
        }
    return out


def params_defensible() -> Dict[str, Any]:
    """
    Baseline calibration using estimated parameters from Egyptian data (2004).
    Sectors: T (textiles+apparel), O (other mfg) — grouping_1.
    Foreign shifters from absorption_2004.json (UNIDO + Comtrade 2004).
    e_ratio_foreign[(j,s)] = E_USD_js / E_USD_EGY_s (size ratios, dimensionless).
    E_foreign and P_foreign are computed dynamically each goods iteration:
      E_foreign[(j,s)] = e_ratio * E_EG_s,  P_foreign[(j,s)] = P_EG_s.
    This gives R_j/R_EG = e_ratio * tau_j^{1-sigma} (small-open-economy formula).
    """
    p: Dict[str, Any] = {}

    # Sets
    p["regions"] = ["Q", "N"]
    p["sectors"] = ["T", "O"]  # textiles, other
    p["dests"] = ["EG", "US", "RW"]

    # Demand elasticities
    # sigma: Imbs & Mejean (2015) literature values
    p["sigma"] = {"T": 6.7, "O": 6.0}
    # eta: upper-tier CES across sectors. eta=1 (Cobb-Douglas) is consistent
    # with IO-table beta_s being constant expenditure shares.
    p["eta"] = 1.0
    # beta: absorption shares from Egypt IO Table 2008 (grouping_1)
    p["beta"] = {"T": 0.052, "O": 0.948}

    # Technology
    # alpha: labor cost shares from Egypt IO Table 2008 (grouping_1)
    p["alpha"] = {"T": 0.186, "O": 0.064}
    # phi_min: normalization (Pareto lower bound sets productivity units)
    p["phi_min"] = {"T": 1.0, "O": 1.0}
    # theta: Pareto shape, estimated via Hill+OLS from customs data (grouping_1)
    p["theta"] = {"T": 6.715, "O": 6.0}

    # Upgrading: phi' = delta_s * phi
    # delta_T from event study (non-US exports, post-period MoM): 1.178
    # delta_O = 1.0 (QIZ applies to textiles only)
    p["delta"] = {"T": 1.178, "O": 1.0}
    p["f_upgrade"] = {"T": 1.8, "O": 2.3}  # fixed cost in labor units
    # Paper-consistent default: upgrading is chosen by profitability (no hard US-only requirement).
    p["upgrade_requires_US"] = False
    # Upgrade choice mode:
    # - "binary": old discrete upgrade/no-upgrade with (delta, f_upgrade)
    # - "continuous": choose intensity u in [0, u_max], with
    #     log(delta) = (psi + psi_comp * 1{comply}) * u
    #   and convex cost in labor units:
    #     fU(u) = f0*1{u>0} + f1*u + 0.5*k*u^2
    p["upgrade_mode"] = "binary"
    p["upgrade_intensity_grid_size"] = 9
    p["upgrade_intensity_max"] = {"T": 2.0, "O": 2.0}
    p["upgrade_psi"] = {"T": 0.12, "O": 0.08}
    p["upgrade_psi_comp"] = {"T": 0.0, "O": 0.0}
    p["upgrade_cost_fixed"] = {"T": 0.0, "O": 0.0}
    p["upgrade_cost_lin"] = {"T": 0.0, "O": 0.0}
    p["upgrade_cost_quad"] = {"T": 4.0, "O": 5.0}
    p["upgrade_cost_comp_mult"] = {"T": 1.0, "O": 1.0}

    # Policy: MFN tariffs (only apply to US, noncompliers)
    # From WTO IDB 2004, value-weighted by Egypt's 2004 exports to US (grouping_1)
    p["t_mfn"] = {"T": 0.096, "O": 0.0095}
    # Tariff treatment:
    # - "iceberg": tariffs are resource wedges (no tariff-revenue rebate endogenized).
    p["tariff_treatment"] = "iceberg"

    # ROO content requirement gamma_s (baseline); can vary in counterfactuals
    p["gamma"] = {"T": 0.105, "O": 0.105}

    # Intermediate prices (exogenous)
    p["p_rw"] = {"T": 1.00, "O": 1.00}
    # Israeli input price premium: 1.20 baseline from ECES 2006 firm survey
    p["p_il"] = {"T": 1.20, "O": 1.20}

    # ROO implementation switches:
    # - "paper": Eq. (17) with denominator gamma^gamma * (1-gamma)^(1-gamma)
    # - "normalized": normalized Cobb-Douglas extension without that denominator
    p["roo_cost_formula"] = "paper"
    # - "US_only": compliance changes only US-destined marginal cost (paper model:
    #   ROO certification is shipment-level, non-US output uses free-sourcing costs)
    p["roo_cost_scope"] = "US_only"
    # Optional extra admin wedge extension: chi(gamma) = 1 + xi*gamma
    p["use_admin_wedge"] = False
    p["xi_admin"] = {"T": 1.0, "O": 0.6}

    # Iceberg trade costs d_{rjs}
    # EK-inverted composite wedges tau_EGY_j from trade_costs.json (grouping_1).
    # Both Q and N assigned the same Egypt-aggregate tau: no region-level export
    # data available to separately identify QIZ vs non-QIZ trade wedges.
    _tc = _load_trade_costs_2004()
    p["d_iceberg"] = {}
    for r in p["regions"]:
        for s in p["sectors"]:
            for j in p["dests"]:
                p["d_iceberg"][(r, j, s)] = _tc[s][j]

    # Fixed costs (labor units)
    p["f_dom"] = {(r, s): (0.9 if s == "T" else 0.8) for r in p["regions"] for s in p["sectors"]}
    p["f_export"] = {}
    for r in p["regions"]:
        for s in p["sectors"]:
            p["f_export"][(r, "US", s)] = (1.2 if s == "T" else 1.7) * (1.0 if r == "Q" else 1.15)
            p["f_export"][(r, "RW", s)] = (1.0 if s == "T" else 1.2) * (1.0 if r == "Q" else 1.10)

    # Entry costs
    p["f_entry"] = {(r, s): (2.0 if s == "T" else 1.7) for r in p["regions"] for s in p["sectors"]}

    # Compliance fixed costs. Paper baseline is deterministic at fC_mean (sigma_C = 0).
    # Keeping sigma_C as a switch preserves backward-compatible heterogeneity if needed.
    # qiz_us_fixed_cost_mode controls how the QIZ fixed cost enters:
    # - "stacked": compliant US exporters pay f_export_US + fC_i
    # - "route_specific": compliant US exporters pay fC_i instead of f_export_US
    p["fC_mean"] = {"T": 0.35, "O": 0.40}
    p["sigma_C"] = {"T": 0.0, "O": 0.0}
    p["qiz_us_fixed_cost_mode"] = "stacked"
    # Optional latent US-incumbency margin. A positive value uses the same eps
    # type that raises QIZ compliance costs to lower the normal US export fixed
    # cost. This captures firms with existing MFN US relationships that are
    # costly to switch into QIZ sourcing/certification.
    p["mfn_us_fixed_cost_discount_sigma"] = {"T": 0.0, "O": 0.0}
    p["mfn_us_fixed_cost_min_mult"] = {"T": 0.25, "O": 0.25}
    p["mfn_us_fixed_cost_max_mult"] = {"T": 4.0, "O": 4.0}

    # Labor / mobility
    # L_total, L_Q, L_N from Egypt LFS 2004 (CAPMAS). Units: persons.
    # kappa = 1.1 placeholder (PPML gravity estimate pending review)
    p["L_total"] = 20_761_200
    p["L_Q_share"] = 0.7387   # L_Q / L_total = 15,336,500 / 20,761,200
    p["kappa"] = 1.1
    # Region-specific amenity/preference shifter in the logit mobility block.
    # It is calibrated in the no-QIZ benchmark to match the observed Q-region share.
    p["mobility_pref"] = {r: 1.0 for r in p["regions"]}
    p["mobility_match_baseline_share"] = True
    p["mobility_target_share_tol"] = 5e-4
    p["mobility_target_max_iter"] = 18
    p["_mobility_pref_calibrated"] = False
    p["_mobility_pref_disable_upgrade"] = None

    # Foreign market size (small open economy).
    # e_ratio_foreign[(j,s)] = E_USD_js / E_USD_EGY_s = ratio of foreign to Egypt absorption.
    # At each iteration, E_foreign[(j,s)] = e_ratio_foreign[(j,s)] * E_EG_s (model units),
    # and P_foreign[(j,s)] = P_EG_s[s] (same price units as domestic).
    # lambda_RW is a temporary fragmentation discount on the pooled non-US market:
    # effective E_RW = lambda_RW * raw pooled E_RW.
    # This ensures R_j/R_EG = (E_j/E_EG) * (tau_j * P_EG / P_EG)^{1-sigma}
    #                       = e_ratio * tau_j^{1-sigma}   (small-open-economy ratio formula).
    # Source: absorption_2004.json grouping_1.
    p["lambda_RW"] = 0.05
    _ab = _load_absorption_2004()
    p["e_ratio_foreign"] = {
        ("US",  "T"): _ab["T"]["E_US"]  / _ab["T"]["E_EGY"],
        ("US",  "O"): _ab["O"]["E_US"]  / _ab["O"]["E_EGY"],
        ("RW",  "T"): _ab["T"]["E_RoW"] / _ab["T"]["E_EGY"],
        ("RW",  "O"): _ab["O"]["E_RoW"] / _ab["O"]["E_EGY"],
    }
    # E_foreign and P_foreign are populated dynamically in solve_goods_block.
    # Initialise them here so firm_best() can read them before the first goods iteration.
    p["E_foreign"] = {k: 1.0 for k in p["e_ratio_foreign"]}
    p["P_foreign"] = {("US", "T"): 1.0, ("US", "O"): 1.0,
                      ("RW", "T"): 1.0, ("RW", "O"): 1.0}

    # Transfer closing small-open economy
    # Exogenous lump-sum transfer used only as a closure term in aggregate income.
    # Under tariff_treatment="iceberg", this is not tariff revenue.
    p["transfer_rule"] = "exogenous_lumpsum"
    p["T_transfer"] = 0.0

    # Compliance feasibility rule used in implementation:
    # if True, a firm can be marked as "complier" only if it serves US.
    p["compliance_requires_US_service"] = True

    # Numerics
    p["n_phi"] = 140        # productivity grid
    p["n_eps"] = 1          # compliance heterogeneity grid (use >1 only if sigma_C>0)
    p["eps_std"] = 3.0      # truncate eps grid to +/- eps_std
    p["goods_tol"] = 2e-4
    p["goods_cycle_tol"] = 5e-3
    p["goods_max_iter"] = 400
    p["outer_tol"] = 5e-4
    p["outer_max_iter"] = 300
    p["outer_step"] = 0.40
    p["outer_cycle_tol"] = 0.02
    p["pair_refine_tol"] = 5e-4
    p["pair_refine_max_iter"] = 3
    p["entry_mass_floor"] = 1e-10
    p["entry_mass_corner_tol"] = 1e-3
    p["entry_mass_cap"] = 1e10

    return p


def params_interior_gamma() -> Dict[str, Any]:
    """
    Alternative parameter preset that delivers an interior ROO-compliance margin,
    so gamma counterfactuals produce visible movements instead of corner solutions.

    This is useful for policy-frontier exploration when placeholder defaults imply
    near-zero compliance at baseline.
    """
    p = params_defensible()
    p["roo_cost_formula"] = "normalized"
    p["roo_cost_scope"] = "US_only"
    p["use_admin_wedge"] = True
    p["xi_admin"] = {"T": 1.2, "O": 0.8}
    # Lower textile compliance fixed cost to encourage interior compliance take-up.
    p["fC_mean"]["T"] = 0.05
    return p


def params_institutional_transparent() -> Dict[str, Any]:
    """
    Institution-forward transparent baseline:
    - US-only ROO scope
    - compliance heterogeneity
    - continuous upgrade intensity with explicit compliance complementarity
    No outcome-targeting tweaks are imposed in this preset.
    """
    p = params_defensible()
    p["roo_cost_formula"] = "normalized"
    p["roo_cost_scope"] = "US_only"
    p["use_admin_wedge"] = True
    p["xi_admin"] = {"T": 1.0, "O": 0.6}
    p["sigma_C"] = {"T": 0.2, "O": 0.15}

    p["upgrade_mode"] = "continuous"
    p["upgrade_requires_US"] = False
    p["upgrade_psi"] = {"T": 0.12, "O": 0.08}
    p["upgrade_psi_comp"] = {"T": 0.06, "O": 0.0}
    p["upgrade_cost_fixed"] = {"T": 0.0, "O": 0.0}
    p["upgrade_cost_lin"] = {"T": 0.0, "O": 0.0}
    p["upgrade_cost_quad"] = {"T": 4.5, "O": 5.5}
    p["upgrade_intensity_grid_size"] = 7
    p["upgrade_intensity_max"] = {"T": 2.0, "O": 2.0}
    # Numerics for continuous-intensity runs.
    p["n_phi"] = 35
    p["n_eps"] = 3
    p["goods_tol"] = 1.0e-3
    p["goods_cycle_tol"] = 3.0e-3
    p["goods_max_iter"] = 180
    p["outer_max_iter"] = 1200
    p["outer_tol"] = 2.0e-3
    p["outer_cycle_tol"] = 1.0e-2
    return p


def params_data_like() -> Dict[str, Any]:
    """
    A tuned preset chosen to reproduce large export gains for complying Q,T firms
    in both US and non-US destinations when comparing QIZ on vs off.

    This is a calibration target, not a structural estimate.
    """
    p = params_interior_gamma()
    p["t_mfn"]["T"] = 0.35
    p["delta"]["T"] = 1.4
    p["f_upgrade"]["T"] = 0.8
    p["fC_mean"]["T"] = 0.06
    p["xi_admin"]["T"] = 0.6
    p["upgrade_requires_US"] = True
    p["roo_cost_formula"] = "normalized"
    p["roo_cost_scope"] = "US_only"
    p["use_admin_wedge"] = True

    # Numerics for this calibration region (stable and fast enough).
    p["n_phi"] = 55
    p["n_eps"] = 3
    p["sigma_C"] = {"T": 0.2, "O": 0.15}
    p["goods_max_iter"] = 170
    p["outer_max_iter"] = 1300
    p["outer_step"] = 0.06
    p["outer_tol"] = 1.5e-3
    p["outer_cycle_tol"] = 7e-3
    p["entry_mass_corner_tol"] = 1e-3
    p["goods_tol"] = 4e-4
    return p


def params_upgrade_complementarity() -> Dict[str, Any]:
    """
    Calibration designed to make upgrading available to all firms, but much more
    likely for Q,T firms that comply with ROO because compliance expands their
    market access and scale.

    This is a targeted mechanism calibration (not a structural estimate).
    """
    p = params_data_like()
    p["upgrade_mode"] = "continuous"
    p["upgrade_requires_US"] = False

    # Make upgrade an interior choice for many non-compliers, but still highly
    # attractive for compliers with US access.
    p["upgrade_psi"]["T"] = 0.14
    p["upgrade_psi_comp"]["T"] = 0.10
    p["upgrade_cost_quad"]["T"] = 4.8
    p["upgrade_intensity_grid_size"] = 7
    p["upgrade_intensity_max"]["T"] = 2.0
    p["t_mfn"]["T"] = 0.35

    # Keep textiles economically relevant so the mechanism is visible.
    p["beta"] = {"T": 0.70, "O": 0.30}
    # e_ratio overrides: these are dimensionless E_j/E_EGY scale factors used in the
    # ratio-based small-open-economy formula.  The original preset used absolute values
    # (500, 350) which are now meaningless; replace with approximate ratios relative to
    # Egypt textile absorption at the new beta=0.70 weighting.
    p["e_ratio_foreign"][("US", "T")] = 9.6     # ≈ 500 / 52  (old scale / avg E_EG_T)
    p["e_ratio_foreign"][("RW", "T")] = 6.7     # ≈ 350 / 52

    # Numerics tuned for stable convergence in this region.
    p["outer_max_iter"] = 1400
    p["outer_tol"] = 2.0e-3
    p["outer_cycle_tol"] = 8.0e-3
    p["n_phi"] = min(p["n_phi"], 35)
    return p


def params_fit_exports_dual_market() -> Dict[str, Any]:
    """
    Constrained fit used for the user's requested targets:
    - MFN tariffs constrained to:
        textiles in [10%, 20%], non-textiles in [0%, 5%]
    - stronger textile upgrading incentives than non-textiles
    - compliers increase exports to both US and non-US destinations
    """
    p = params_institutional_transparent()

    # Tariff constraints requested by user.
    p["t_mfn"]["T"] = 0.15
    p["t_mfn"]["O"] = 0.03

    # Keep transparent mechanism assumptions.
    p["roo_cost_scope"] = "US_only"
    p["roo_cost_formula"] = "normalized"
    p["upgrade_mode"] = "continuous"
    p["upgrade_requires_US"] = True

    # Textile-specific upgrade incentive stronger than non-textile.
    p["upgrade_psi"]["T"] = 0.12
    p["upgrade_psi"]["O"] = 0.08
    p["upgrade_psi_comp"]["T"] = 0.06
    p["upgrade_psi_comp"]["O"] = 0.0
    p["upgrade_cost_quad"]["T"] = 4.5
    p["upgrade_cost_quad"]["O"] = 5.5

    return p


def params_fit_us_aggregate_positive() -> Dict[str, Any]:
    """
    Parameter-only variant (no new model blocks) chosen to produce:
    - positive aggregate US export response,
    - positive Q-region US export response,
    while preserving positive complier US and non-US export responses.
    """
    p = params_fit_exports_dual_market()

    # Mild textile-demand shift only; no O-sector QIZ shortcut.
    p["e_ratio_foreign"][("US", "T")] *= 1.25

    # Keep tariffs within user-requested bounds.
    p["t_mfn"]["T"] = 0.15
    p["t_mfn"]["O"] = 0.05

    return p


def params_expected_direction_base() -> Dict[str, Any]:
    """
    Benchmark designed for the mechanism the project wants to study:
    QIZ use should improve US access, induce upgrading, and spill over into
    non-US exports for the same treated textile firms.

    This preset keeps the estimated/data-backed primitives from params_defensible()
    but switches on the mechanism blocks needed for that directional exercise:
    - normalized ROO cost formula
    - partial compliance via heterogeneity
    - continuous upgrading with a compliance complementarity
    - upgrading feasible only for US-serving firms
    """
    p = params_defensible()

    p["roo_cost_formula"] = "normalized"
    p["roo_cost_scope"] = "US_only"
    p["use_admin_wedge"] = True

    p["n_eps"] = 5
    p["sigma_C"] = {"T": 0.20, "O": 0.0}

    p["upgrade_mode"] = "continuous"
    p["upgrade_requires_US"] = True
    p["upgrade_psi"] = {"T": 0.10, "O": 0.08}
    p["upgrade_psi_comp"] = {"T": 0.06, "O": 0.0}
    p["upgrade_cost_fixed"] = {"T": 0.0, "O": 0.0}
    p["upgrade_cost_lin"] = {"T": 0.0, "O": 0.0}
    p["upgrade_cost_quad"] = {"T": 4.5, "O": 5.0}
    p["upgrade_intensity_grid_size"] = 7
    p["upgrade_intensity_max"] = {"T": 2.0, "O": 2.0}

    # Calibration-oriented numerics for a stable benchmark solve.
    p["n_phi"] = 35
    p["goods_max_iter"] = 180
    p["outer_max_iter"] = 1000
    p["outer_tol"] = 2.0e-3
    p["outer_cycle_tol"] = 8.0e-3
    return p


# -----------------------------
# Numerical helpers
# -----------------------------

def pareto_grid(phi_min: float, theta: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Midpoint rule in u in (0,1): phi(u)=phi_min*(1-u)^(-1/theta)."""
    u = (np.arange(n) + 0.5) / n
    u = np.clip(u, 1e-12, 1-1e-12)
    phi = phi_min * (1.0 - u) ** (-1.0 / theta)
    w = np.full(n, 1.0 / n)
    return phi, w

def normal_grid(n: int, a: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Midpoint-rule quadrature for eps ~ N(0,1) over [-a, a].
    Uses uniform spacing h and weights w_i = pdf(x_i)*h, normalized to sum to 1
    so that E[f(eps)] ~ sum_i w_i * f(x_i) approximates the truncated expectation.
    """
    x = -a + (np.arange(n) + 0.5) * (2.0 * a / n)  # true midpoint rule (no endpoints)
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    h = 2.0 * a / n
    w = pdf * h
    w = w / w.sum()  # normalize so weights sum to 1 over the truncated support
    return x, w

def ces_price_from_power(P_pow: float, sigma: float) -> float:
    """Given P_pow = ∫ p^{1-sigma}, return P = P_pow^(1/(1-sigma))."""
    if P_pow <= 0:
        return 1e6
    return P_pow ** (1.0 / (1.0 - sigma))

def ces_aggregate_price(P_by_s: Dict[str, float], beta: Dict[str, float], eta: float) -> float:
    """Aggregate across sectors: P = [sum_s beta_s * P_s^(1-eta)]^(1/(1-eta))."""
    if abs(eta - 1.0) < 1e-12:
        return float(np.exp(sum(beta[s] * np.log(P_by_s[s]) for s in beta)))
    power = 1.0 - eta
    inside = sum(beta[s] * (P_by_s[s] ** power) for s in beta)
    return float(inside ** (1.0 / power))

def sector_expenditures(Y: float, P_by_s: Dict[str, float], P_agg: float, beta: Dict[str, float], eta: float) -> Dict[str, float]:
    """CES upper tier: E_s = beta_s * (P_s/P)^(1-eta) * Y."""
    return {s: beta[s] * ((P_by_s[s] / P_agg) ** (1.0 - eta)) * Y for s in beta}

def admin_wedge(gamma: float, xi: float) -> float:
    return 1.0 + xi * gamma

def safe_log_change(a: float, b: float, floor: float = 1.0e-12) -> float:
    """log(a)-log(b) with floors; returns NaN for non-finite inputs."""
    if (not np.isfinite(a)) or (not np.isfinite(b)):
        return np.nan
    return float(np.log(max(a, floor)) - np.log(max(b, floor)))

def safe_pct_change(a: float, b: float, min_denom: float = 1.0e-12, clip: float | None = None) -> float:
    """Percent change 100*(a-b)/|b| with denominator floor and optional clipping."""
    if (not np.isfinite(a)) or (not np.isfinite(b)):
        return np.nan
    out = 100.0 * (a - b) / max(abs(b), min_denom)
    if clip is not None:
        out = float(np.clip(out, -clip, clip))
    return float(out)

def cmix_paper(p_il: float, p_rw: float, gamma: float) -> float:
    """
    Paper Eq. (17):
      c_mix = p_il^gamma * p_rw^(1-gamma) / [gamma^gamma * (1-gamma)^(1-gamma)].
    """
    if gamma <= 0:
        return p_rw
    if gamma >= 1:
        return p_il
    denom = (gamma ** gamma) * ((1.0 - gamma) ** (1.0 - gamma))
    return (p_il ** gamma) * (p_rw ** (1.0 - gamma)) / denom

def cmix_normalized(p_il: float, p_rw: float, gamma: float) -> float:
    """Normalized Cobb–Douglas => unit cost = p_il^gamma * p_rw^(1-gamma)."""
    if gamma <= 0:
        return p_rw
    if gamma >= 1:
        return p_il
    return (p_il ** gamma) * (p_rw ** (1.0 - gamma))


def invalidate_benchmark_calibrations(p: Dict[str, Any]) -> Dict[str, Any]:
    """Mark cached benchmark calibrations as stale after structural parameter changes."""
    p["_mobility_pref_calibrated"] = False
    p["_mobility_pref_disable_upgrade"] = None
    p.pop("_mobility_pref_target_share", None)
    p.pop("_mobility_pref_matched_share", None)
    return p


def restrict_to_regions(p: Dict[str, Any], keep_regions: List[str]) -> Dict[str, Any]:
    """
    Restrict the model to a subset of regions already present in the parameter
    dictionary. When only one region remains, the mobility block is degenerate,
    so mobility-share targeting is switched off and the region's labor share is
    fixed at one.
    """
    keep = list(keep_regions)
    if not keep:
        raise ValueError("keep_regions must contain at least one region.")

    current = list(p.get("regions", []))
    missing = [r for r in keep if r not in current]
    if missing:
        raise ValueError(f"Unknown regions requested: {missing}")

    keep_set = set(keep)
    p["regions"] = keep

    if "d_iceberg" in p:
        p["d_iceberg"] = {
            (r, j, s): v
            for (r, j, s), v in p["d_iceberg"].items()
            if r in keep_set
        }
    if "f_dom" in p:
        p["f_dom"] = {
            (r, s): v
            for (r, s), v in p["f_dom"].items()
            if r in keep_set
        }
    if "f_export" in p:
        p["f_export"] = {
            (r, j, s): v
            for (r, j, s), v in p["f_export"].items()
            if r in keep_set
        }
    if "f_entry" in p:
        p["f_entry"] = {
            (r, s): v
            for (r, s), v in p["f_entry"].items()
            if r in keep_set
        }

    old_pref = p.get("mobility_pref", {})
    p["mobility_pref"] = {r: float(old_pref.get(r, 1.0)) for r in keep}

    if len(keep) == 1:
        only = keep[0]
        p["mobility_pref"] = {only: 1.0}
        p["mobility_match_baseline_share"] = False
        if only == "Q":
            p["L_Q_share"] = 1.0

    return invalidate_benchmark_calibrations(p)


def _set_two_region_mobility_pref(p: Dict[str, Any], q_over_n: float) -> None:
    if set(p["regions"]) != {"Q", "N"}:
        raise NotImplementedError("Mobility-preference calibration currently assumes regions {Q, N}.")
    p["mobility_pref"] = {r: 1.0 for r in p["regions"]}
    p["mobility_pref"]["Q"] = float(q_over_n)
    p["mobility_pref"]["N"] = 1.0


def calibrate_mobility_pref_to_baseline_share(
    p: Dict[str, Any],
    disable_upgrade: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Choose the Q-vs-N mobility shifter so the no-QIZ equilibrium matches the
    observed benchmark labor share in Q.
    """
    target = float(p["L_Q_share"])
    tol = float(p.get("mobility_target_share_tol", 5e-4))
    max_iter = int(p.get("mobility_target_max_iter", 18))

    def eval_trial(log_ratio: float, warm_start: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any], float]:
        _set_two_region_mobility_pref(p, float(np.exp(log_ratio)))
        sol = solve_equilibrium(
            p,
            qiz_on=False,
            disable_upgrade=disable_upgrade,
            initial_state=warm_start,
            verbose=False,
        )
        share_q = sol["Ls"]["Q"] / max(p["L_total"], 1e-12)
        return share_q - target, sol, share_q

    gap_0, sol_0, share_0 = eval_trial(0.0)
    best_gap = gap_0
    best_log = 0.0
    best_sol = sol_0
    best_share = share_0

    if gap_0 <= 0.0:
        lo, gap_lo, sol_lo, share_lo = 0.0, gap_0, sol_0, share_0
        hi = gap_hi = sol_hi = share_hi = None
        cur_log = 0.0
        cur_sol = sol_0
        step = 0.5
        for _ in range(12):
            trial_log = cur_log + step
            try:
                gap_t, sol_t, share_t = eval_trial(trial_log, warm_start=cur_sol)
            except RuntimeError:
                step *= 0.5
                if step < 0.05:
                    raise RuntimeError("Could not bracket the target Q labor share when calibrating mobility_pref.")
                continue
            if abs(gap_t) < abs(best_gap):
                best_gap = gap_t
                best_log = trial_log
                best_sol = sol_t
                best_share = share_t
            if gap_t >= 0.0:
                hi, gap_hi, sol_hi, share_hi = trial_log, gap_t, sol_t, share_t
                break
            lo, gap_lo, sol_lo, share_lo = trial_log, gap_t, sol_t, share_t
            cur_log = trial_log
            cur_sol = sol_t
            step *= 1.5
        if hi is None or gap_hi is None or sol_hi is None or share_hi is None:
            raise RuntimeError("Could not bracket the target Q labor share when calibrating mobility_pref.")
    else:
        hi, gap_hi, sol_hi, share_hi = 0.0, gap_0, sol_0, share_0
        lo = gap_lo = sol_lo = share_lo = None
        cur_log = 0.0
        cur_sol = sol_0
        step = 0.5
        for _ in range(12):
            trial_log = cur_log - step
            try:
                gap_t, sol_t, share_t = eval_trial(trial_log, warm_start=cur_sol)
            except RuntimeError:
                step *= 0.5
                if step < 0.05:
                    raise RuntimeError("Could not bracket the target Q labor share when calibrating mobility_pref.")
                continue
            if abs(gap_t) < abs(best_gap):
                best_gap = gap_t
                best_log = trial_log
                best_sol = sol_t
                best_share = share_t
            if gap_t <= 0.0:
                lo, gap_lo, sol_lo, share_lo = trial_log, gap_t, sol_t, share_t
                break
            hi, gap_hi, sol_hi, share_hi = trial_log, gap_t, sol_t, share_t
            cur_log = trial_log
            cur_sol = sol_t
            step *= 1.5
        if lo is None or gap_lo is None or sol_lo is None or share_lo is None:
            raise RuntimeError("Could not bracket the target Q labor share when calibrating mobility_pref.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gap_mid, sol_mid, share_mid = eval_trial(mid, warm_start=best_sol)

        if abs(gap_mid) < abs(best_gap):
            best_gap = gap_mid
            best_log = mid
            best_sol = sol_mid
            best_share = share_mid

        if abs(gap_mid) < tol:
            break

        if gap_mid > 0.0:
            hi, gap_hi, sol_hi, share_hi = mid, gap_mid, sol_mid, share_mid
        else:
            lo, gap_lo, sol_lo, share_lo = mid, gap_mid, sol_mid, share_mid

    _set_two_region_mobility_pref(p, float(np.exp(best_log)))
    p["_mobility_pref_calibrated"] = True
    p["_mobility_pref_disable_upgrade"] = bool(disable_upgrade)
    p["_mobility_pref_target_share"] = target
    p["_mobility_pref_matched_share"] = best_share

    if verbose:
        print(
            "mobility calibration:",
            f"target={target:.4f}",
            f"matched={best_share:.4f}",
            f"pref_Q={p['mobility_pref']['Q']:.4f}",
        )

    return {
        "off": best_sol,
        "target_share": target,
        "matched_share": best_share,
        "mobility_pref": dict(p["mobility_pref"]),
    }


def ensure_benchmark_calibrations(
    p: Dict[str, Any],
    disable_upgrade: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run benchmark-side calibrations that should not be embedded in every GE solve."""
    regions = list(p.get("regions", []))
    if len(regions) <= 1:
        if len(regions) == 1:
            p["mobility_pref"] = {regions[0]: 1.0}
        p["_mobility_pref_calibrated"] = True
        p["_mobility_pref_disable_upgrade"] = bool(disable_upgrade)
        p["_mobility_pref_target_share"] = 1.0 if regions == ["Q"] else None
        p["_mobility_pref_matched_share"] = 1.0 if regions == ["Q"] else None
        return {
            "off": None,
            "target_share": p.get("_mobility_pref_target_share"),
            "matched_share": p.get("_mobility_pref_matched_share"),
            "mobility_pref": dict(p.get("mobility_pref", {})),
        }

    if not p.get("mobility_match_baseline_share", True):
        p["_mobility_pref_calibrated"] = True
        p["_mobility_pref_disable_upgrade"] = bool(disable_upgrade)
        return {
            "off": None,
            "target_share": p.get("L_Q_share"),
            "matched_share": None,
            "mobility_pref": dict(p.get("mobility_pref", {})),
        }

    if p.get("_mobility_pref_calibrated", False) and p.get("_mobility_pref_disable_upgrade") == bool(disable_upgrade):
        return {
            "off": None,
            "target_share": p.get("_mobility_pref_target_share", p.get("L_Q_share")),
            "matched_share": p.get("_mobility_pref_matched_share"),
            "mobility_pref": dict(p.get("mobility_pref", {})),
        }

    return calibrate_mobility_pref_to_baseline_share(
        p,
        disable_upgrade=disable_upgrade,
        verbose=verbose,
    )


# -----------------------------
# Firm problem
# -----------------------------

def firm_best(phi: float, eps: float, r: str, s: str,
              w_r: float, P_EG_s: float, E_EG_s: float,
              p: Dict[str, Any],
              gamma_override: Dict[str, float] | None,
              qiz_on: bool,
              disable_upgrade: bool) -> Dict[str, Any]:
    """
    Choose compliance (Q-only) and upgrade, given CES demand and fixed costs.
    - ROO implementation (cost formula + destination scope) is controlled by parameters.
    - Optionally, upgrading can be constrained to US-serving firms.
    """
    sigma = p["sigma"][s]
    mu = sigma / (sigma - 1.0)
    alpha = p["alpha"][s]
    denom = (alpha ** alpha) * ((1.0 - alpha) ** (1.0 - alpha))

    # policy / gamma
    gamma_s = (gamma_override or p["gamma"])[s]
    t_mfn = p["t_mfn"][s]

    # intermediate unit costs
    cN = p["p_rw"][s]
    roo_formula = p.get("roo_cost_formula", "paper")
    if roo_formula == "paper":
        cmix = cmix_paper(p["p_il"][s], p["p_rw"][s], gamma_s)
    elif roo_formula == "normalized":
        cmix = cmix_normalized(p["p_il"][s], p["p_rw"][s], gamma_s)
    else:
        raise ValueError(f"Unknown roo_cost_formula={roo_formula!r}")
    admin = admin_wedge(gamma_s, p["xi_admin"][s]) if p.get("use_admin_wedge", False) else 1.0
    cC = admin * cmix
    roo_scope = p.get("roo_cost_scope", "all_destinations")

    # foreign shifters
    E_US = p["E_foreign"][("US", s)]
    E_RW = p["E_foreign"][("RW", s)]
    P_US = p["P_foreign"][("US", s)]
    P_RW = p["P_foreign"][("RW", s)]

    # fixed costs
    f_dom = p["f_dom"][(r, s)]
    f_US = p["f_export"][(r, "US", s)]
    f_RW = p["f_export"][(r, "RW", s)]
    fC_i = p["fC_mean"][s] * np.exp(p["sigma_C"][s] * eps)
    mfn_sigma_cfg = p.get("mfn_us_fixed_cost_discount_sigma", {})
    mfn_min_cfg = p.get("mfn_us_fixed_cost_min_mult", {})
    mfn_max_cfg = p.get("mfn_us_fixed_cost_max_mult", {})
    mfn_sigma = float(mfn_sigma_cfg.get(s, 0.0) if isinstance(mfn_sigma_cfg, dict) else mfn_sigma_cfg)
    mfn_min = float(mfn_min_cfg.get(s, 0.25) if isinstance(mfn_min_cfg, dict) else mfn_min_cfg)
    mfn_max = float(mfn_max_cfg.get(s, 4.0) if isinstance(mfn_max_cfg, dict) else mfn_max_cfg)
    f_US_i = f_US * float(np.clip(np.exp(-mfn_sigma * eps), mfn_min, mfn_max))
    comp_upgrade_cost_cfg = p.get("upgrade_cost_comp_mult", {})
    comp_upgrade_cost_mult = float(
        comp_upgrade_cost_cfg.get(s, 1.0)
        if isinstance(comp_upgrade_cost_cfg, dict)
        else comp_upgrade_cost_cfg
    )
    qiz_us_fixed_cost_mode = p.get("qiz_us_fixed_cost_mode", "stacked")
    if qiz_us_fixed_cost_mode not in {"stacked", "route_specific"}:
        raise ValueError(f"Unknown qiz_us_fixed_cost_mode={qiz_us_fixed_cost_mode!r}")

    # strategy space
    upgrade_mode = p.get("upgrade_mode", "binary")
    if upgrade_mode not in {"binary", "continuous"}:
        raise ValueError(f"Unknown upgrade_mode={upgrade_mode!r}")

    compliance_choices: List[bool] = [False]
    if (r == "Q") and qiz_on:
        compliance_choices.append(True)

    if disable_upgrade:
        u_grid = [0.0]
    elif upgrade_mode == "binary":
        u_grid = [0.0, 1.0]
    else:
        n_u = max(2, int(p.get("upgrade_intensity_grid_size", 9)))
        u_max = p["upgrade_intensity_max"][s]
        u_grid = list(np.linspace(0.0, float(u_max), n_u))

    best_out = None
    best_profit = -1e18

    for comply in compliance_choices:
        qiz_compliance_choice = comply and (r == "Q") and qiz_on

        if qiz_compliance_choice:
            if roo_scope == "all_destinations":
                c_base = cC
                c_us = cC
            elif roo_scope == "US_only":
                c_base = cN
                c_us = cC
            else:
                raise ValueError(f"Unknown roo_cost_scope={roo_scope!r}")
        else:
            c_base = cN
            c_us = cN

        tau_US = p["d_iceberg"][(r, "US", s)] * (1.0 if (comply and r == "Q" and qiz_on) else (1.0 + t_mfn))
        tau_EG = p["d_iceberg"][(r, "EG", s)]
        tau_RW = p["d_iceberg"][(r, "RW", s)]

        f_US_route = (
            fC_i
            if (comply and (r == "Q") and qiz_on and qiz_us_fixed_cost_mode == "route_specific")
            else f_US_i
        )

        def evaluate_market(delta_eff: float, us_fixed_cost: float) -> Dict[str, Any]:
            mc_base_ = (1.0 / (phi * delta_eff)) * ((w_r ** alpha) * (c_base ** (1.0 - alpha))) / denom
            mc_US_ = (1.0 / (phi * delta_eff)) * ((w_r ** alpha) * (c_us ** (1.0 - alpha))) / denom

            p_EG_ = mu * (tau_EG * mc_base_)
            p_RW_ = mu * (tau_RW * mc_base_)
            p_US_ = mu * (tau_US * mc_US_)

            R_EG_ = E_EG_s * ((p_EG_ / P_EG_s) ** (1.0 - sigma))
            R_US_ = E_US * ((p_US_ / P_US) ** (1.0 - sigma))
            R_RW_ = E_RW * ((p_RW_ / P_RW) ** (1.0 - sigma))

            pi_EG_ = R_EG_ / sigma
            pi_US_ = R_US_ / sigma
            pi_RW_ = R_RW_ / sigma

            serve_EG_ = (pi_EG_ >= w_r * f_dom)
            serve_US_ = (pi_US_ >= w_r * us_fixed_cost)
            serve_RW_ = (pi_RW_ >= w_r * f_RW)
            return {
                "R_EG": R_EG_, "R_US": R_US_, "R_RW": R_RW_,
                "serve_EG": serve_EG_, "serve_US": serve_US_, "serve_RW": serve_RW_,
            }

        for u in u_grid:
            if upgrade_mode == "binary":
                eff_upgrade = bool(u > 0.5)
                upgrade_intensity = 1.0 if eff_upgrade else 0.0
                eff_delta = p["delta"][s] if eff_upgrade else 1.0
                upgrade_labor = p["f_upgrade"][s] if eff_upgrade else 0.0
                if eff_upgrade and qiz_compliance_choice:
                    upgrade_labor *= comp_upgrade_cost_mult
            else:
                upgrade_intensity = float(u)
                eff_upgrade = bool(upgrade_intensity > 1e-12)
                comp_bonus = 1.0 if qiz_compliance_choice else 0.0
                slope = p["upgrade_psi"][s] + p["upgrade_psi_comp"][s] * comp_bonus
                eff_delta = float(np.exp(slope * upgrade_intensity))
                if eff_upgrade:
                    upgrade_labor = (
                        p["upgrade_cost_fixed"][s]
                        + p["upgrade_cost_lin"][s] * upgrade_intensity
                        + 0.5 * p["upgrade_cost_quad"][s] * (upgrade_intensity ** 2)
                    )
                    if qiz_compliance_choice:
                        upgrade_labor *= comp_upgrade_cost_mult
                else:
                    upgrade_labor = 0.0

            out = evaluate_market(eff_delta, f_US_route)

            if eff_upgrade and p["upgrade_requires_US"] and (not out["serve_US"]):
                eff_upgrade = False
                upgrade_intensity = 0.0
                eff_delta = 1.0
                upgrade_labor = 0.0
                out = evaluate_market(eff_delta, f_US_route)

            req_us_for_comp = p.get("compliance_requires_US_service", True)
            if req_us_for_comp:
                eff_comply = comply and (r == "Q") and qiz_on and out["serve_US"]
                if comply and (r == "Q") and qiz_on and (not out["serve_US"]):
                    continue
            else:
                eff_comply = comply and (r == "Q") and qiz_on

            profit = 0.0
            if out["serve_EG"]:
                profit += (out["R_EG"] / sigma - w_r * f_dom)
            if out["serve_US"]:
                profit += (out["R_US"] / sigma - w_r * f_US_route)
            if out["serve_RW"]:
                profit += (out["R_RW"] / sigma - w_r * f_RW)

            if eff_comply and qiz_us_fixed_cost_mode == "stacked":
                profit -= w_r * fC_i
            profit -= w_r * upgrade_labor

            if profit < 0 or (not out["serve_EG"] and not out["serve_US"] and not out["serve_RW"]):
                profit = 0.0
                out["serve_EG"] = False
                out["serve_US"] = False
                out["serve_RW"] = False
                eff_upgrade = False
                upgrade_intensity = 0.0
                eff_delta = 1.0
                eff_comply = False
                upgrade_labor = 0.0

            if profit > best_profit:
                best_profit = profit
                effective_phi = phi * eff_delta

                total_R = (
                    (out["R_EG"] if out["serve_EG"] else 0.0)
                    + (out["R_US"] if out["serve_US"] else 0.0)
                    + (out["R_RW"] if out["serve_RW"] else 0.0)
                )
                var_cost = (sigma - 1.0) / sigma * total_R
                is_active = out["serve_EG"] or out["serve_US"] or out["serve_RW"]
                labor_var = alpha * var_cost / w_r if is_active else 0.0

                labor_fixed = 0.0
                if out["serve_EG"]:
                    labor_fixed += f_dom
                if out["serve_US"]:
                    labor_fixed += f_US_route
                if out["serve_RW"]:
                    labor_fixed += f_RW
                if eff_comply and qiz_us_fixed_cost_mode == "stacked":
                    labor_fixed += fC_i
                labor_fixed += upgrade_labor

                best_out = {
                    "profit": profit,
                    "serve": {
                        "EG": out["serve_EG"],
                        "US": out["serve_US"],
                        "RW": out["serve_RW"],
                    },
                    "compliance": bool(eff_comply),
                    "us_route": (
                        "QIZ"
                        if (out["serve_US"] and eff_comply)
                        else ("MFN" if out["serve_US"] else None)
                    ),
                    "us_fixed_cost_paid": float(f_US_route if out["serve_US"] else 0.0),
                    "upgrade": bool(eff_upgrade),
                    "upgrade_intensity": float(upgrade_intensity),
                    "upgrade_delta": float(eff_delta),
                    "effective_phi": effective_phi,
                    "R": {
                        "EG": (out["R_EG"] if out["serve_EG"] else 0.0),
                        "US": (out["R_US"] if out["serve_US"] else 0.0),
                        "RW": (out["R_RW"] if out["serve_RW"] else 0.0),
                    },
                    "labor_var": labor_var,
                    "labor_fixed": labor_fixed,
                }

    assert best_out is not None
    return best_out

# -----------------------------
# Goods block: solve P_EG,s, E_EG,s given wages & entry masses
# -----------------------------

def solve_goods_block(w: Dict[str, float], M: Dict[Tuple[str,str], float],
                      p: Dict[str, Any],
                      gamma_override: Dict[str,float] | None,
                      qiz_on: bool,
                      disable_upgrade: bool,
                      P_init: Dict[str, float] | None = None) -> Dict[str, Any]:
    regions = p["regions"]
    sectors = p["sectors"]
    beta = p["beta"]
    eta = p["eta"]
    kappa = p["kappa"]

    # grids (precompute per sector)
    phi_grid = {}
    w_phi = {}
    for s in sectors:
        phi_grid[s], w_phi[s] = pareto_grid(p["phi_min"][s], p["theta"][s], p["n_phi"])
    eps_grid, w_eps = normal_grid(p["n_eps"], a=p["eps_std"])

    # initialize sector price indices (warm start from prior outer iteration if provided)
    P_EG_s = {}
    for s in sectors:
        if P_init is None:
            P_EG_s[s] = 1.0
        else:
            guess = float(P_init.get(s, 1.0))
            P_EG_s[s] = guess if guess > 0 else 1.0

    def evaluate_at_prices(P_curr: Dict[str, float]) -> Tuple[Dict[str, float], float, float, Dict[str, float], Dict[str, float], Dict[Tuple[str, str], Dict[str, float]]]:
        P_EG = ces_aggregate_price(P_curr, beta, eta)

        # labor supply from mobility
        mobility_pref = p.get("mobility_pref", {r: 1.0 for r in regions})
        numer = {r: mobility_pref.get(r, 1.0) * (w[r] / P_EG) ** kappa for r in regions}
        denom_mob = sum(numer.values())
        Ls = {r: (numer[r] / denom_mob) * p["L_total"] for r in regions}

        Y = sum(w[r] * Ls[r] for r in regions) + p["T_transfer"]
        E_EG_s = sector_expenditures(Y, P_curr, P_EG, beta, eta)

        # Foreign market shifters in model units:
        #   E_foreign[(j,s)] = e_ratio[(j,s)] * E_EG_s[s]
        #   P_foreign[(j,s)] = P_EG_s[s]   (same price units as domestic)
        # This gives R_j/R_EG = e_ratio * tau_j^{1-sigma} (small-open-economy formula).
        e_ratio = p.get("e_ratio_foreign", {})
        lambda_rw = float(p.get("lambda_RW", 1.0))
        for (j, s), ratio in e_ratio.items():
            scale = lambda_rw if j == "RW" else 1.0
            p["E_foreign"][(j, s)] = scale * ratio * E_EG_s[s]
            p["P_foreign"][(j, s)] = P_curr[s]

        # update sector price indices from varieties serving EG
        P_new: Dict[str, float] = {}
        cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        for s in sectors:
            sigma = p["sigma"][s]
            P_pow = 0.0

            for r in regions:
                contrib = 0.0
                Ep = Elv = Elf = Eact = Eus = Erw = Eany = Edom = Ec = Eu = Eu_int = 0.0
                ER_EG = ER_US = ER_RW = 0.0

                for phi, wp_ in zip(phi_grid[s], w_phi[s]):
                    for eps, we_ in zip(eps_grid, w_eps):
                        wt = wp_ * we_
                        best = firm_best(
                            phi, eps, r, s, w[r], P_curr[s], E_EG_s[s], p,
                            gamma_override=gamma_override,
                            qiz_on=qiz_on,
                            disable_upgrade=disable_upgrade
                        )

                        Ep += wt * best["profit"]
                        Elv += wt * best["labor_var"]
                        Elf += wt * best["labor_fixed"]

                        active = 1.0 if any(best["serve"].values()) else 0.0
                        any_export = 1.0 if (best["serve"]["US"] or best["serve"]["RW"]) else 0.0
                        dom_only = 1.0 if (best["serve"]["EG"] and (not best["serve"]["US"]) and (not best["serve"]["RW"])) else 0.0
                        Eact += wt * active
                        Eus += wt * (1.0 if best["serve"]["US"] else 0.0)
                        Erw += wt * (1.0 if best["serve"]["RW"] else 0.0)
                        Eany += wt * any_export
                        Edom += wt * dom_only
                        Ec += wt * (1.0 if best["compliance"] else 0.0)
                        Eu += wt * (1.0 if best["upgrade"] else 0.0)
                        Eu_int += wt * best.get("upgrade_intensity", 0.0)
                        ER_EG += wt * best["R"]["EG"]
                        ER_US += wt * best["R"]["US"]
                        ER_RW += wt * best["R"]["RW"]

                        if best["serve"]["EG"] and E_EG_s[s] > 0:
                            # Use CES identity R = E * (p/P)^(1-sigma) to recover p^(1-sigma)
                            contrib += wt * (best["R"]["EG"] / E_EG_s[s]) * (P_curr[s] ** (1.0 - sigma))

                P_pow += M[(r, s)] * contrib
                cache[(r, s)] = {
                    "E_profit": Ep,
                    "E_lvar": Elv,
                    "E_lfix": Elf,
                    "E_active": Eact,
                    "E_US": Eus,
                    "E_RW": Erw,
                    "E_any_export": Eany,
                    "E_dom_only": Edom,
                    "E_comp": Ec,
                    "E_up": Eu,
                    "E_up_int": Eu_int,
                    "E_R_EG": ER_EG,
                    "E_R_US": ER_US,
                    "E_R_RW": ER_RW,
                }

            P_new[s] = ces_price_from_power(P_pow, sigma)

        return P_new, P_EG, Y, E_EG_s, Ls, cache

    hist_logs: List[Dict[str, float]] = []
    last_diff = np.inf
    for _ in range(p["goods_max_iter"]):
        P_new, _, _, _, _, _ = evaluate_at_prices(P_EG_s)
        last_diff = max(abs(np.log(P_new[s]) - np.log(P_EG_s[s])) for s in sectors)

        damp = 0.55
        for s in sectors:
            P_EG_s[s] = float(np.exp((1.0 - damp) * np.log(P_EG_s[s]) + damp * np.log(P_new[s])))

        hist_logs.append({s: float(np.log(P_EG_s[s])) for s in sectors})

        if last_diff < p["goods_tol"]:
            _, P_EG, Y, E_EG_s, Ls, cache = evaluate_at_prices(P_EG_s)
            return {"P_EG_s": P_EG_s, "P_EG": P_EG, "Y": Y, "E_EG_s": E_EG_s, "Ls": Ls, "cache": cache}

        if len(hist_logs) >= 3:
            # 2-cycle guard for discrete policy kinks on finite grids.
            cycle_gap = max(abs(hist_logs[-1][s] - hist_logs[-3][s]) for s in sectors)
            if cycle_gap < p.get("goods_cycle_tol", 5.0 * p["goods_tol"]):
                for s in sectors:
                    P_EG_s[s] = float(np.exp(0.5 * (hist_logs[-1][s] + hist_logs[-2][s])))
                _, P_EG, Y, E_EG_s, Ls, cache = evaluate_at_prices(P_EG_s)
                return {"P_EG_s": P_EG_s, "P_EG": P_EG, "Y": Y, "E_EG_s": E_EG_s, "Ls": Ls, "cache": cache}

    raise RuntimeError(f"Goods block did not converge (last diff={last_diff:.2e}).")


# -----------------------------
# Outer GE solver
# -----------------------------

def solve_equilibrium(p: Dict[str, Any],
                      qiz_on: bool = True,
                      gamma_override: Dict[str,float] | None = None,
                      disable_upgrade: bool = False,
                      initial_state: Dict[str, Any] | None = None,
                      verbose: bool = True) -> Dict[str, Any]:
    if p.get("tariff_treatment", "iceberg") != "iceberg":
        raise NotImplementedError("Only tariff_treatment='iceberg' is implemented.")
    if p.get("transfer_rule", "exogenous_lumpsum") != "exogenous_lumpsum":
        raise NotImplementedError("Only transfer_rule='exogenous_lumpsum' is implemented.")

    regions = p["regions"]
    sectors = p["sectors"]

    # wages / entry masses (optionally warm-started from a previous equilibrium)
    if initial_state is None:
        w = {r: 1.0 for r in regions}
        M = {(r,s): 1.0 for r in regions for s in sectors}
        P_guess = None
    else:
        w = {r: float(initial_state["w"][r]) for r in regions}
        M = {(r, s): float(initial_state["M"][(r, s)]) for r in regions for s in sectors}
        P_guess = {s: float(initial_state["goods"]["P_EG_s"][s]) for s in sectors}

    def finalize_solution(goods: Dict[str, Any], cache: Dict[Tuple[str, str], Dict[str, float]],
                          Ld: Dict[str, float], Ls: Dict[str, float],
                          w_state: Dict[str, float], M_state: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
        W = goods["Y"] / goods["P_EG"]
        moments = {}
        for rr in regions:
            for ss in sectors:
                mom = cache[(rr, ss)]
                active = max(mom["E_active"], 1e-12)
                moments[(rr, ss)] = {
                    "active_share": mom["E_active"],
                    "any_export_share_among_active": mom["E_any_export"]/active,
                    "domestic_only_share_among_active": mom["E_dom_only"]/active,
                    "US_export_share_among_active": mom["E_US"]/active,
                    "RW_export_share_among_active": mom["E_RW"]/active,
                    "compliance_share_among_active": mom["E_comp"]/active if rr == "Q" else 0.0,
                    "compliance_share_among_US_exporters": (
                        mom["E_comp"] / max(mom["E_US"], 1e-12)
                        if rr == "Q" else 0.0
                    ),
                    "upgrade_share_among_active": mom["E_up"]/active,
                    "upgrade_intensity_among_active": mom.get("E_up_int", 0.0)/active,
                }
        return {
            "w": dict(w_state), "M": dict(M_state),
            "goods": goods,
            "Ld": Ld, "Ls": Ls,
            "welfare": W,
            "moments": moments,
            "qiz_on": qiz_on,
            "gamma": (gamma_override or p["gamma"]),
            "disable_upgrade": disable_upgrade
        }

    state_logs: List[Dict[str, float]] = []
    cycle_tol = p.get("outer_cycle_tol", 5.0 * p["outer_tol"])
    cycle_accept_metric = p.get("outer_cycle_accept_metric", max(5.0 * p["outer_tol"], cycle_tol))
    M_prev: Dict[Tuple[str, str], float] = {}

    for it in range(p["outer_max_iter"]):
        # Discard goods-block warm start when M is still changing rapidly
        # (large M jump → stale P_guess causes goods block divergence).
        if M_prev and P_guess is not None:
            m_jump = max(
                abs(np.log(max(M[(r,s)],1e-300)) - np.log(max(M_prev[(r,s)],1e-300)))
                for r in regions for s in sectors
            )
            if m_jump > 1.5:
                P_guess = None
        M_prev = dict(M)
        try:
            goods = solve_goods_block(
                w, M, p, gamma_override,
                qiz_on=qiz_on,
                disable_upgrade=disable_upgrade,
                P_init=P_guess
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Goods block failure at outer iteration {it}. "
                f"State: w={w}, M={M}"
            ) from exc
        P_guess = goods["P_EG_s"]
        cache = goods["cache"]
        Ls = goods["Ls"]

        # labor demand per region
        Ld = {r: 0.0 for r in regions}
        for r in regions:
            for s in sectors:
                Ld[r] += M[(r,s)] * (p["f_entry"][(r,s)] + cache[(r,s)]["E_lvar"] + cache[(r,s)]["E_lfix"])

        labor_ratio = {r: max(Ld[r],1e-12)/max(Ls[r],1e-12) for r in regions}

        # free entry conditions per (r,s)
        entry_ratio = {}
        for r in regions:
            for s in sectors:
                entry_cost = w[r] * p["f_entry"][(r,s)]
                entry_ratio[(r,s)] = max(cache[(r,s)]["E_profit"], 1e-12) / max(entry_cost, 1e-12)

        m_floor = p.get("entry_mass_floor", 1e-12)
        corner_mass_tol = max(10.0 * m_floor, p.get("entry_mass_corner_tol", 10.0 * m_floor))
        entry_gaps = []
        for r in regions:
            for s in sectors:
                x = np.log(entry_ratio[(r, s)])
                if M[(r, s)] > corner_mass_tol:
                    # Interior free-entry condition: E[pi] = w*f_entry.
                    entry_gaps.append(abs(x))
                else:
                    # Corner condition at zero mass: E[pi] <= w*f_entry.
                    entry_gaps.append(max(x, 0.0))

        # Convergence metric: free-entry gaps only.
        # Labor market clearing is not enforced as a convergence condition because
        # in a numeraire-normalized Melitz model wages are indeterminate in level;
        # the labor_ratio updates the *relative* wage (w_Q/w_N via logit mobility)
        # and the wage normalization step below keeps the level at 1.
        metric = max(entry_gaps)

        if verbose and (it % 40 == 0 or metric < p["outer_tol"]):
            wage_bits = " ".join(f"w[{r}]={w[r]:.3f}" for r in regions)
            labor_bits = " ".join(f"Ld/Ls[{r}]={labor_ratio[r]:.3f}" for r in regions)
            print(f"it={it:4d} metric={metric:.2e} {wage_bits} {labor_bits}")

        if metric < p["outer_tol"]:
            return finalize_solution(goods, cache, Ld, Ls, w, M)

        state_log = {f"w_{r}": float(np.log(w[r])) for r in regions}
        for r in regions:
            for s in sectors:
                state_log[f"M_{r}_{s}"] = float(np.log(max(M[(r, s)], 1e-300)))
        state_logs.append(state_log)
        if len(state_logs) >= 3:
            cycle_gap = max(abs(state_logs[-1][k] - state_logs[-3][k]) for k in state_log)
            if cycle_gap < cycle_tol and metric < cycle_accept_metric:
                if verbose:
                    print(f"outer-cycle accept at it={it}: metric={metric:.2e}, cycle_gap={cycle_gap:.2e}")
                return finalize_solution(goods, cache, Ld, Ls, w, M)

        step = p["outer_step"]
        # update wages and entry masses
        for r in regions:
            w[r] *= float(np.exp(step * np.log(labor_ratio[r])))
            w[r] = float(np.clip(w[r], 1e-4, 1e4))
        # Wage-level normalization: keep labor-weighted average wage = 1.
        # In a single-factor Melitz model wages are a numeraire; without this
        # the absolute wage level drifts (all real allocations are wage-homogeneous)
        # causing the solver to spiral toward w→0 or w→∞ when Ld/Ls ≠ 1.
        w_avg = sum(Ls[r] * w[r] for r in regions) / max(sum(Ls.values()), 1e-12)
        if w_avg > 1e-12:
            for r in regions:
                w[r] = float(w[r] / w_avg)
        m_cap = p.get("entry_mass_cap", 1e12)
        for r in regions:
            for s in sectors:
                if M[(r, s)] <= corner_mass_tol and entry_ratio[(r, s)] < 1.0:
                    M[(r, s)] = float(m_floor)
                    continue
                M[(r, s)] *= float(np.exp(step * np.log(entry_ratio[(r, s)])))
                M[(r, s)] = float(np.clip(M[(r, s)], m_floor, m_cap))

    raise RuntimeError("Outer GE did not converge.")


def summarize_trade(sol: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate trade outcomes from cached firm expectations.
    Returns totals by (region,destination), exporter masses, and average revenue/exporter.
    """
    regions = p["regions"]
    sectors = p["sectors"]
    cache = sol["goods"]["cache"]
    M = sol["M"]

    exports: Dict[Tuple[str, str], float] = {}
    exporters: Dict[Tuple[str, str], float] = {}
    avg_rev_per_exporter: Dict[Tuple[str, str], float] = {}

    for r in regions:
        for j in ["US", "RW"]:
            X = 0.0
            N = 0.0
            for s in sectors:
                mom = cache[(r, s)]
                if j == "US":
                    X += M[(r, s)] * mom["E_R_US"]
                    N += M[(r, s)] * mom["E_US"]
                else:
                    X += M[(r, s)] * mom["E_R_RW"]
                    N += M[(r, s)] * mom["E_RW"]
            exports[(r, j)] = X
            exporters[(r, j)] = N
            avg_rev_per_exporter[(r, j)] = X / max(N, 1e-12)

    return {
        "exports": exports,
        "exporter_masses": exporters,
        "avg_revenue_per_exporter": avg_rev_per_exporter,
    }


def summarize_qt_groups(sol: Dict[str, Any], p: Dict[str, Any], qiz_on: bool = True,
                        disable_upgrade: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Q,T decomposition by on-equilibrium group membership:
    - compliers: firms that choose compliance in Q,T
    - noncompliers: active Q,T firms that do not comply
    Returns shares, upgrade rates, and average revenues by destination.
    """
    s, r = "T", "Q"
    phi_grid, w_phi = pareto_grid(p["phi_min"][s], p["theta"][s], p["n_phi"])
    eps_grid, w_eps = normal_grid(p["n_eps"], a=p["eps_std"])

    w_r = sol["w"][r]
    P_EG_s = sol["goods"]["P_EG_s"][s]
    E_EG_s = sol["goods"]["E_EG_s"][s]

    stats = {
        "comp": {"mass": 0.0, "up": 0.0, "u": 0.0, "R_US": 0.0, "R_RW": 0.0, "R_tot": 0.0},
        "noncomp": {"mass": 0.0, "up": 0.0, "u": 0.0, "R_US": 0.0, "R_RW": 0.0, "R_tot": 0.0},
    }

    for phi, wp_ in zip(phi_grid, w_phi):
        for eps, we_ in zip(eps_grid, w_eps):
            wt = wp_ * we_
            best = firm_best(
                phi, eps, r, s, w_r, P_EG_s, E_EG_s, p,
                gamma_override=None,
                qiz_on=qiz_on,
                disable_upgrade=disable_upgrade
            )
            if not any(best["serve"].values()):
                continue

            g = "comp" if best["compliance"] else "noncomp"
            stats[g]["mass"] += wt
            stats[g]["up"] += wt * (1.0 if best["upgrade"] else 0.0)
            stats[g]["u"] += wt * best.get("upgrade_intensity", 0.0)
            stats[g]["R_US"] += wt * best["R"]["US"]
            stats[g]["R_RW"] += wt * best["R"]["RW"]
            stats[g]["R_tot"] += wt * (best["R"]["EG"] + best["R"]["US"] + best["R"]["RW"])

    active_mass = stats["comp"]["mass"] + stats["noncomp"]["mass"]
    out: Dict[str, Dict[str, float]] = {}
    for g in ["comp", "noncomp"]:
        m = max(stats[g]["mass"], 1e-12)
        out[g] = {
            "share_among_active_QT": stats[g]["mass"] / max(active_mass, 1e-12),
            "upgrade_rate": stats[g]["up"] / m,
            "avg_upgrade_intensity": stats[g]["u"] / m,
            "avg_R_US": stats[g]["R_US"] / m,
            "avg_R_RW": stats[g]["R_RW"] / m,
            "avg_R_total": stats[g]["R_tot"] / m,
        }
    return out


def summarize_trade_by_sector(sol: Dict[str, Any], p: Dict[str, Any]) -> Dict[Tuple[str, str, str], float]:
    """
    Sector-destination exports: X_{r,s,j} for j in {US,RW}.
    """
    out: Dict[Tuple[str, str, str], float] = {}
    for r in p["regions"]:
        for s in p["sectors"]:
            mom = sol["goods"]["cache"][(r, s)]
            mrs = sol["M"][(r, s)]
            out[(r, s, "US")] = mrs * mom["E_R_US"]
            out[(r, s, "RW")] = mrs * mom["E_R_RW"]
    return out


def summarize_qt_same_type_changes(on: Dict[str, Any], off: Dict[str, Any], p: Dict[str, Any],
                                   disable_upgrade: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Micro decomposition for Q,T.
    Grouping is done using on-state (QIZ on) firm types:
    - comp: active firms that comply in on-state
    - noncomp: active firms that do not comply in on-state
    For each group, evaluate on/off average revenues and upgrade intensity.
    """
    s, r = "T", "Q"
    phi_grid, w_phi = pareto_grid(p["phi_min"][s], p["theta"][s], p["n_phi"])
    eps_grid, w_eps = normal_grid(p["n_eps"], a=p["eps_std"])

    stats = {
        "comp": {"mass": 0.0, "u_on": 0.0, "u_off": 0.0, "US_on": 0.0, "US_off": 0.0, "RW_on": 0.0, "RW_off": 0.0, "TOT_on": 0.0, "TOT_off": 0.0},
        "noncomp": {"mass": 0.0, "u_on": 0.0, "u_off": 0.0, "US_on": 0.0, "US_off": 0.0, "RW_on": 0.0, "RW_off": 0.0, "TOT_on": 0.0, "TOT_off": 0.0},
    }

    for phi, wp_ in zip(phi_grid, w_phi):
        for eps, we_ in zip(eps_grid, w_eps):
            wt = wp_ * we_
            b_on = firm_best(
                phi, eps, r, s,
                on["w"][r], on["goods"]["P_EG_s"][s], on["goods"]["E_EG_s"][s],
                p, gamma_override=None, qiz_on=True, disable_upgrade=disable_upgrade
            )
            if not any(b_on["serve"].values()):
                continue

            grp = "comp" if b_on["compliance"] else "noncomp"
            b_off = firm_best(
                phi, eps, r, s,
                off["w"][r], off["goods"]["P_EG_s"][s], off["goods"]["E_EG_s"][s],
                p, gamma_override=None, qiz_on=False, disable_upgrade=disable_upgrade
            )

            stats[grp]["mass"] += wt
            stats[grp]["u_on"] += wt * b_on.get("upgrade_intensity", 0.0)
            stats[grp]["u_off"] += wt * b_off.get("upgrade_intensity", 0.0)
            stats[grp]["US_on"] += wt * b_on["R"]["US"]
            stats[grp]["US_off"] += wt * b_off["R"]["US"]
            stats[grp]["RW_on"] += wt * b_on["R"]["RW"]
            stats[grp]["RW_off"] += wt * b_off["R"]["RW"]
            stats[grp]["TOT_on"] += wt * (b_on["R"]["EG"] + b_on["R"]["US"] + b_on["R"]["RW"])
            stats[grp]["TOT_off"] += wt * (b_off["R"]["EG"] + b_off["R"]["US"] + b_off["R"]["RW"])

    def pct(a: float, b: float) -> float:
        raw = 100.0 * (a - b) / max(abs(b), 1e-12)
        return float(np.clip(raw, -500.0, 500.0))

    out: Dict[str, Dict[str, float]] = {}
    mass_tot = stats["comp"]["mass"] + stats["noncomp"]["mass"]
    min_group_mass = 1.0e-6
    for grp in ["comp", "noncomp"]:
        m = max(stats[grp]["mass"], 1e-12)
        if stats[grp]["mass"] < min_group_mass:
            us_pct = np.nan
            rw_pct = np.nan
            tot_pct = np.nan
        else:
            us_pct = pct(stats[grp]["US_on"], stats[grp]["US_off"])
            rw_pct = pct(stats[grp]["RW_on"], stats[grp]["RW_off"])
            tot_pct = pct(stats[grp]["TOT_on"], stats[grp]["TOT_off"])
        out[grp] = {
            "share_among_active_QT": stats[grp]["mass"] / max(mass_tot, 1e-12),
            "group_mass": stats[grp]["mass"],
            "defined": bool(stats[grp]["mass"] >= min_group_mass),
            "u_on": stats[grp]["u_on"] / m,
            "u_off": stats[grp]["u_off"] / m,
            "US_on_level": stats[grp]["US_on"],
            "US_off_level": stats[grp]["US_off"],
            "RW_on_level": stats[grp]["RW_on"],
            "RW_off_level": stats[grp]["RW_off"],
            "TOT_on_level": stats[grp]["TOT_on"],
            "TOT_off_level": stats[grp]["TOT_off"],
            "US_pct": us_pct,
            "RW_pct": rw_pct,
            "TOT_pct": tot_pct,
        }
    return out


def collect_stylized_moments(result: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, float]:
    """
    Compact moment bundle for transparent calibration.
    """
    on = result["on"]
    off = result["off"]
    dec = result["decomposition"]
    disable_upgrade = bool(on.get("disable_upgrade", False))

    tr_on = summarize_trade(on, p)
    tr_off = summarize_trade(off, p)
    sec_on = summarize_trade_by_sector(on, p)
    sec_off = summarize_trade_by_sector(off, p)
    qt_grp = summarize_qt_groups(on, p, qiz_on=True, disable_upgrade=disable_upgrade)
    qt_same = summarize_qt_same_type_changes(on, off, p, disable_upgrade=disable_upgrade)

    def pct_safe(a: float, b: float, ref: float) -> float:
        # Avoid explosive percentages when a sector's baseline is tiny.
        denom = max(abs(b), 0.01 * abs(ref), 1e-12)
        return 100.0 * (a - b) / denom

    x_all_on_us = sum(tr_on["exports"][(r, "US")] for r in p["regions"])
    x_all_off_us = sum(tr_off["exports"][(r, "US")] for r in p["regions"])
    x_all_on_rw = sum(tr_on["exports"][(r, "RW")] for r in p["regions"])
    x_all_off_rw = sum(tr_off["exports"][(r, "RW")] for r in p["regions"])

    x_qt_on_us = sec_on[("Q", "T", "US")]
    x_qt_off_us = sec_off[("Q", "T", "US")]
    x_qt_on_rw = sec_on[("Q", "T", "RW")]
    x_qt_off_rw = sec_off[("Q", "T", "RW")]

    def rs_employment(sol: Dict[str, Any], r: str, s: str) -> float:
        c = sol["goods"]["cache"][(r, s)]
        return sol["M"][(r, s)] * (p["f_entry"][(r, s)] + c["E_lvar"] + c["E_lfix"])

    def rs_production(sol: Dict[str, Any], r: str, s: str) -> float:
        c = sol["goods"]["cache"][(r, s)]
        return sol["M"][(r, s)] * (c["E_R_EG"] + c["E_R_US"] + c["E_R_RW"])

    emp_qt_on = rs_employment(on, "Q", "T")
    emp_qt_off = rs_employment(off, "Q", "T")
    prod_qt_on = rs_production(on, "Q", "T")
    prod_qt_off = rs_production(off, "Q", "T")
    rw_q_on = on["w"]["Q"] / max(on["goods"]["P_EG"], 1e-12)
    rw_q_off = off["w"]["Q"] / max(off["goods"]["P_EG"], 1e-12)
    rw_n_on = on["w"]["N"] / max(on["goods"]["P_EG"], 1e-12)
    rw_n_off = off["w"]["N"] / max(off["goods"]["P_EG"], 1e-12)

    comp_defined = bool(qt_same["comp"]["defined"])
    comp_us_pct = float(np.clip(qt_same["comp"]["US_pct"], -500.0, 500.0)) if comp_defined else np.nan
    comp_rw_pct = float(np.clip(qt_same["comp"]["RW_pct"], -500.0, 500.0)) if comp_defined else np.nan
    comp_tot_pct = float(np.clip(qt_same["comp"]["TOT_pct"], -500.0, 500.0)) if comp_defined else np.nan

    return {
        "micro_comp_us_pct": comp_us_pct,
        "micro_comp_rw_pct": comp_rw_pct,
        "micro_comp_tot_pct": comp_tot_pct,
        "micro_upgrade_gap_u": qt_grp["comp"]["avg_upgrade_intensity"] - qt_grp["noncomp"]["avg_upgrade_intensity"],
        "agg_q_t_us_pct": float(np.clip(pct_safe(x_qt_on_us, x_qt_off_us, tr_off["exports"][("Q", "US")]), -500.0, 500.0)),
        "agg_q_t_rw_pct": float(np.clip(pct_safe(x_qt_on_rw, x_qt_off_rw, tr_off["exports"][("Q", "RW")]), -500.0, 500.0)),
        "agg_q_total_us_pct": float(np.clip(dec[("Q", "US")]["pct_change"], -500.0, 500.0)),
        "agg_q_total_rw_pct": float(np.clip(dec[("Q", "RW")]["pct_change"], -500.0, 500.0)),
        "agg_total_us_pct": safe_pct_change(x_all_on_us, x_all_off_us, clip=500.0),
        "agg_total_rw_pct": safe_pct_change(x_all_on_rw, x_all_off_rw, clip=500.0),
        "q_t_active_share": on["moments"][("Q", "T")]["active_share"],
        "q_t_comp_share_active": on["moments"][("Q", "T")]["compliance_share_among_active"],
        # User-requested "increase-only" targets.
        "target_emp_qt_pct": safe_pct_change(emp_qt_on, emp_qt_off, clip=500.0),
        "target_prod_qt_pct": safe_pct_change(prod_qt_on, prod_qt_off, clip=500.0),
        "target_real_wage_q_pct": safe_pct_change(rw_q_on, rw_q_off, clip=500.0),
        "target_comp_us_pct": comp_us_pct,
        "target_comp_rw_pct": comp_rw_pct,
        # Tier 1: core identification moments (micro mechanism).
        "tier1_comp_us_logchg": safe_log_change(qt_same["comp"]["US_on_level"], qt_same["comp"]["US_off_level"]) if comp_defined else np.nan,
        "tier1_comp_rw_logchg": safe_log_change(qt_same["comp"]["RW_on_level"], qt_same["comp"]["RW_off_level"]) if comp_defined else np.nan,
        "tier1_comp_share_active": on["moments"][("Q", "T")]["compliance_share_among_active"],
        "tier1_comp_us_dlevel": (qt_same["comp"]["US_on_level"] - qt_same["comp"]["US_off_level"]) if comp_defined else np.nan,
        "tier1_comp_rw_dlevel": (qt_same["comp"]["RW_on_level"] - qt_same["comp"]["RW_off_level"]) if comp_defined else np.nan,
        # Tier 2: treated-region firm outcomes.
        "tier2_emp_qt_logchg": safe_log_change(emp_qt_on, emp_qt_off),
        "tier2_prod_qt_logchg": safe_log_change(prod_qt_on, prod_qt_off),
        "tier2_real_wage_q_logchg": safe_log_change(rw_q_on, rw_q_off),
        "tier2_emp_qt_dlevel": float(emp_qt_on - emp_qt_off),
        "tier2_prod_qt_dlevel": float(prod_qt_on - prod_qt_off),
        "tier2_real_wage_q_dlevel": float(rw_q_on - rw_q_off),
        # Tier 3: aggregate discipline moments.
        "tier3_qt_us_logchg": safe_log_change(x_qt_on_us, x_qt_off_us),
        "tier3_qt_rw_logchg": safe_log_change(x_qt_on_rw, x_qt_off_rw),
        "tier3_total_us_logchg": safe_log_change(x_all_on_us, x_all_off_us),
        "tier3_total_rw_logchg": safe_log_change(x_all_on_rw, x_all_off_rw),
        "tier3_real_wage_n_logchg": safe_log_change(rw_n_on, rw_n_off),
        "tier3_welfare_logchg": safe_log_change(on["welfare"], off["welfare"]),
        "tier3_qt_us_dlevel": float(x_qt_on_us - x_qt_off_us),
        "tier3_qt_rw_dlevel": float(x_qt_on_rw - x_qt_off_rw),
        "tier3_total_us_dlevel": float(x_all_on_us - x_all_off_us),
        "tier3_total_rw_dlevel": float(x_all_on_rw - x_all_off_rw),
    }


def default_tiered_moment_spec() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Tiered moment targets and weights:
    - Tier 1 identifies the mechanism.
    - Tier 2 matches treated-firm outcome responses.
    - Tier 3 disciplines aggregate implications (lower weight).
    Targets are set as minima in log-change space for an increase-only fit.
    """
    targets = {
        # Tier 1
        "tier1_comp_us_logchg": float(np.log(1.5)),
        "tier1_comp_rw_logchg": float(np.log(1.1)),
        "tier1_comp_share_active": 0.05,
        # Tier 2
        "tier2_emp_qt_logchg": 0.0,
        "tier2_prod_qt_logchg": 0.0,
        "tier2_real_wage_q_logchg": 0.0,
        # Tier 3
        "tier3_qt_us_logchg": 0.0,
        "tier3_qt_rw_logchg": 0.0,
        "tier3_total_us_logchg": 0.0,
        "tier3_welfare_logchg": 0.0,
    }
    weights = {
        # Tier 1 (highest)
        "tier1_comp_us_logchg": 5.0,
        "tier1_comp_rw_logchg": 5.0,
        "tier1_comp_share_active": 3.0,
        # Tier 2 (medium)
        "tier2_emp_qt_logchg": 2.0,
        "tier2_prod_qt_logchg": 2.0,
        "tier2_real_wage_q_logchg": 1.5,
        # Tier 3 (discipline)
        "tier3_qt_us_logchg": 1.0,
        "tier3_qt_rw_logchg": 1.0,
        "tier3_total_us_logchg": 0.8,
        "tier3_welfare_logchg": 0.8,
    }
    return targets, weights


def baseline_snapshot_from_result(p: Dict[str, Any],
                                  result: Dict[str, Any],
                                  preset: str) -> Dict[str, Any]:
    moms = collect_stylized_moments(result, p)
    on = result["on"]
    off = result["off"]
    tr_on = summarize_trade(on, p)
    tr_off = summarize_trade(off, p)
    return {
        "meta": {
            "preset": preset,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "schema_version": 1,
        },
        "parameters": {
            "t_mfn": p["t_mfn"],
            "gamma": p["gamma"],
            "upgrade_mode": p.get("upgrade_mode", "binary"),
            "upgrade_psi": p.get("upgrade_psi", {}),
            "upgrade_psi_comp": p.get("upgrade_psi_comp", {}),
            "upgrade_cost_quad": p.get("upgrade_cost_quad", {}),
            "fC_mean": p["fC_mean"],
            "roo_cost_scope": p.get("roo_cost_scope", "all_destinations"),
            "roo_cost_formula": p.get("roo_cost_formula", "paper"),
        },
        "equilibrium": {
            "welfare_on": on["welfare"],
            "welfare_off": off["welfare"],
            "wages_on": on["w"],
            "wages_off": off["w"],
            "employment_shares_on": {r: on["Ls"][r] / p["L_total"] for r in p["regions"]},
            "employment_shares_off": {r: off["Ls"][r] / p["L_total"] for r in p["regions"]},
            "exports_on_Q_US": tr_on["exports"][("Q", "US")],
            "exports_off_Q_US": tr_off["exports"][("Q", "US")],
            "exports_on_Q_RW": tr_on["exports"][("Q", "RW")],
            "exports_off_Q_RW": tr_off["exports"][("Q", "RW")],
        },
        "moments": moms,
    }


def write_baseline_snapshot(path: str, snapshot: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)


def check_baseline_snapshot(path: str, current: Dict[str, Any], tol: float = 1.0e-8) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        frozen = json.load(f)
    cur_m = current.get("moments", {})
    old_m = frozen.get("moments", {})
    keys = sorted(set(cur_m.keys()).intersection(old_m.keys()))
    diffs = []
    for k in keys:
        a = cur_m[k]
        b = old_m[k]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        d = abs(float(a) - float(b))
        if d > tol:
            diffs.append((k, float(a), float(b), d))
    return {"frozen": frozen, "diffs": diffs}


def print_baseline_table(snapshot: Dict[str, Any]) -> None:
    eq = snapshot["equilibrium"]
    print("\n=== Frozen Baseline Table ===")
    print(f"Preset: {snapshot['meta']['preset']}")
    print(f"Created (UTC): {snapshot['meta']['created_utc']}")
    print(f"Welfare on/off: {eq['welfare_on']:.6f} / {eq['welfare_off']:.6f}")
    print(f"Wages on: {eq['wages_on']}")
    print(f"Wages off: {eq['wages_off']}")
    print(f"Employment shares on: {eq['employment_shares_on']}")
    print(f"Employment shares off: {eq['employment_shares_off']}")
    print(
        "Q exports US on/off:",
        f"{eq['exports_on_Q_US']:.6f} / {eq['exports_off_Q_US']:.6f}",
    )
    print(
        "Q exports RW on/off:",
        f"{eq['exports_on_Q_RW']:.6f} / {eq['exports_off_Q_RW']:.6f}",
    )


def _apply_calibration_knob(p: Dict[str, Any], knob: str, value: float):
    if knob == "upgrade_psi_T":
        p["upgrade_psi"]["T"] = float(value)
    elif knob == "upgrade_psi_comp_T":
        p["upgrade_psi_comp"]["T"] = float(value)
    elif knob == "upgrade_cost_quad_T":
        p["upgrade_cost_quad"]["T"] = float(value)
    elif knob == "t_mfn_T":
        p["t_mfn"]["T"] = float(value)
    elif knob == "fC_mean_T":
        p["fC_mean"]["T"] = float(value)
    else:
        raise ValueError(f"Unknown calibration knob: {knob}")
    invalidate_benchmark_calibrations(p)


def fit_to_stylized_moments(base_p: Dict[str, Any],
                            targets: Dict[str, float],
                            weights: Dict[str, float],
                            grid: Dict[str, List[float]],
                            minimum_only: bool = False,
                            verbose: bool = True) -> Dict[str, Any]:
    """
    Transparent coarse-grid calibration:
    - minimize weighted squared distance to declared targets
    - no hidden post-hoc tweaking
    """
    knobs = list(grid.keys())
    values = [grid[k] for k in knobs]
    n_total = int(np.prod([len(v) for v in values])) if values else 0
    tried = 0
    best: Dict[str, Any] | None = None

    for combo in itertools.product(*values):
        tried += 1
        p_try = copy.deepcopy(base_p)
        for k, v in zip(knobs, combo):
            _apply_calibration_knob(p_try, k, v)

        try:
            res = compare_qiz_on_off(p_try, disable_upgrade=False)
        except RuntimeError:
            continue

        moms = collect_stylized_moments(res, p_try)
        loss = 0.0
        for mk, mt in targets.items():
            mv = moms.get(mk, np.nan)
            if not np.isfinite(mv):
                loss += 1e9
                continue
            if minimum_only:
                shortfall = max(0.0, mt - mv)
                loss += weights.get(mk, 1.0) * (shortfall ** 2)
            else:
                loss += weights.get(mk, 1.0) * ((mv - mt) ** 2)

        cand = {
            "loss": loss,
            "params": {k: v for k, v in zip(knobs, combo)},
            "moments": moms,
            "result": res,
            "p": p_try,
        }
        if (best is None) or (cand["loss"] < best["loss"]):
            best = cand
            if verbose:
                print(f"fit update: tried={tried}/{n_total} loss={loss:.4f} params={cand['params']}")

    if best is None:
        raise RuntimeError("No converged calibration candidate found in grid.")
    return best


def _solution_log_gap(a: Dict[str, Any], b: Dict[str, Any], floor: float = 1.0e-12) -> float:
    gaps: List[float] = []
    for r in a["w"]:
        gaps.append(abs(np.log(max(a["w"][r], floor)) - np.log(max(b["w"][r], floor))))
    for key in a["M"]:
        gaps.append(abs(np.log(max(a["M"][key], floor)) - np.log(max(b["M"][key], floor))))
    for s in a["goods"]["P_EG_s"]:
        gaps.append(abs(np.log(max(a["goods"]["P_EG_s"][s], floor)) - np.log(max(b["goods"]["P_EG_s"][s], floor))))
    for r in a["Ls"]:
        gaps.append(abs(np.log(max(a["Ls"][r], floor)) - np.log(max(b["Ls"][r], floor))))
    gaps.append(abs(np.log(max(a["welfare"], floor)) - np.log(max(b["welfare"], floor))))
    return max(gaps) if gaps else 0.0


def solve_policy_pair(
    p: Dict[str, Any],
    disable_upgrade: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve the no-QIZ and QIZ equilibria with a common benchmark protocol:
    1. calibrate regional mobility to the no-QIZ geography,
    2. solve off then on,
    3. refine with off/on warm-start passes to reduce path dependence.
    """
    calibration = ensure_benchmark_calibrations(
        p,
        disable_upgrade=disable_upgrade,
        verbose=verbose,
    )
    calibration_public = {k: v for k, v in calibration.items() if k != "off"}
    off = calibration.get("off")
    if off is None:
        off = solve_equilibrium(p, qiz_on=False, disable_upgrade=disable_upgrade, verbose=False)
    on = solve_equilibrium(
        p,
        qiz_on=True,
        disable_upgrade=disable_upgrade,
        initial_state=off,
        verbose=False,
    )

    pair_gap = float("nan")
    pair_iterations = 0
    tol = float(p.get("pair_refine_tol", p.get("outer_tol", 5e-4)))
    max_iter = int(p.get("pair_refine_max_iter", 3))

    for it in range(max_iter):
        off_next = solve_equilibrium(
            p,
            qiz_on=False,
            disable_upgrade=disable_upgrade,
            initial_state=on,
            verbose=False,
        )
        on_next = solve_equilibrium(
            p,
            qiz_on=True,
            disable_upgrade=disable_upgrade,
            initial_state=off_next,
            verbose=False,
        )
        pair_gap = max(_solution_log_gap(off_next, off), _solution_log_gap(on_next, on))
        off, on = off_next, on_next
        pair_iterations = it + 1

        if verbose:
            print(f"pair refine it={pair_iterations} gap={pair_gap:.2e}")

        if pair_gap < tol:
            break

    return {
        "off": off,
        "on": on,
        "pair_gap": pair_gap,
        "pair_iterations": pair_iterations,
        "benchmark_calibration": calibration_public,
    }


def compare_qiz_on_off(p: Dict[str, Any], disable_upgrade: bool = False) -> Dict[str, Any]:
    """
    Solve QIZ-on and QIZ-off equilibria and report trade decomposition:
    X = N * xbar where N is exporter mass and xbar is average exports per exporter.
    """
    pair = solve_policy_pair(p, disable_upgrade=disable_upgrade, verbose=False)
    on = pair["on"]
    off = pair["off"]

    tr_on = summarize_trade(on, p)
    tr_off = summarize_trade(off, p)

    dec: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key in tr_on["exports"]:
        X1 = tr_on["exports"][key]
        X0 = tr_off["exports"][key]
        N1 = tr_on["exporter_masses"][key]
        N0 = tr_off["exporter_masses"][key]
        x1 = tr_on["avg_revenue_per_exporter"][key]
        x0 = tr_off["avg_revenue_per_exporter"][key]

        dX = X1 - X0
        ext = (N1 - N0) * x0
        inten = N0 * (x1 - x0)
        interact = (N1 - N0) * (x1 - x0)
        pct = 100.0 * dX / max(abs(X0), 1e-12)

        dec[key] = {
            "on": X1,
            "off": X0,
            "delta": dX,
            "pct_change": pct,
            "extensive_component": ext,
            "intensive_component": inten,
            "interaction_component": interact,
        }

    return {
        "on": on,
        "off": off,
        "trade_on": tr_on,
        "trade_off": tr_off,
        "decomposition": dec,
        "pair_gap": pair["pair_gap"],
        "pair_iterations": pair["pair_iterations"],
        "benchmark_calibration": pair["benchmark_calibration"],
    }


# -----------------------------
# Counterfactual routines
# -----------------------------

def counterfactual_shutdown_productivity(p: Dict[str, Any], baseline: Dict[str, Any] | None = None) -> Dict[str, Dict[str, Any]]:
    """Compare baseline vs delta=1 (productivity channel off)."""
    if baseline is None:
        base = solve_policy_pair(p, disable_upgrade=False, verbose=False)["on"]
    else:
        base = baseline

    p_off = copy.deepcopy(p)  # deep copy to avoid mutating nested dicts in original p
    if p_off.get("upgrade_mode", "binary") == "continuous":
        p_off["upgrade_psi"] = {s: 0.0 for s in p["sectors"]}
        p_off["upgrade_psi_comp"] = {s: 0.0 for s in p["sectors"]}
    else:
        p_off["delta"] = {s: 1.0 for s in p["sectors"]}
    # (Alternatively: make f_upgrade huge) -- keep as-is here since delta=1 is simplest.
    off = solve_equilibrium(p_off, qiz_on=True, disable_upgrade=False, initial_state=base, verbose=False)

    return {"baseline": base, "no_productivity": off}

def counterfactual_gamma_path(p: Dict[str, Any], sector: str, gamma_list: List[float],
                              start_state: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Solve GE for each gamma value for a given sector, keeping everything else fixed."""
    out = []
    prev = start_state if start_state is not None else solve_policy_pair(p, disable_upgrade=False, verbose=False)["on"]
    for g in gamma_list:
        gamma_override = dict(p["gamma"])
        gamma_override[sector] = float(g)
        sol = solve_equilibrium(
            p,
            qiz_on=True,
            gamma_override=gamma_override,
            disable_upgrade=False,
            initial_state=prev,
            verbose=False
        )
        prev = sol
        out.append({
            "gamma": float(g),
            "welfare": sol["welfare"],
            "wQ": sol["w"]["Q"],
            "wN": sol["w"]["N"],
            "LQ_share": sol["Ls"]["Q"]/p["L_total"],
            "comp_QT": sol["moments"][("Q", sector)]["compliance_share_among_active"],
            "USexp_QT": sol["moments"][("Q", sector)]["US_export_share_among_active"],
            "RWexp_QT": sol["moments"][("Q", sector)]["RW_export_share_among_active"],
            "upgrade_QT": sol["moments"][("Q", sector)]["upgrade_share_among_active"],
            "upgrade_intensity_QT": sol["moments"][("Q", sector)].get("upgrade_intensity_among_active", 0.0),
        })
    return out


# -----------------------------
# Main
# -----------------------------

def print_key(sol: Dict[str, Any], label: str, p: Dict[str, Any]):
    print(f"\n=== {label} ===")
    print("Welfare (Y/P):", sol["welfare"])
    print("Wages:", sol["w"])
    print("Employment shares:", {r: sol["Ls"][r]/p["L_total"] for r in sol["Ls"]})
    print("Domestic P_EG:", sol["goods"]["P_EG"])
    print("Domestic sector prices:", sol["goods"]["P_EG_s"])
    print("Selected moments (Q,T):", sol["moments"][("Q","T")])


def print_assumption_report(p: Dict[str, Any], preset_name: str):
    """
    Print a transparent assumptions block so runs are auditable.
    """
    print("\n=== Assumption Report ===")
    print(f"Preset: {preset_name}")
    print("Paper-core switches:")
    print(f"  roo_cost_formula={p['roo_cost_formula']}")
    print(f"  roo_cost_scope={p['roo_cost_scope']}")
    print(f"  use_admin_wedge={p['use_admin_wedge']}")
    print(f"  upgrade_requires_US={p['upgrade_requires_US']}")
    print(f"  compliance_requires_US_service={p.get('compliance_requires_US_service', True)}")
    print(f"  tariff_treatment={p.get('tariff_treatment', 'iceberg')}")
    print(f"  transfer_rule={p.get('transfer_rule', 'exogenous_lumpsum')}")
    print(f"  upgrade_mode={p.get('upgrade_mode', 'binary')}")
    print(f"  lambda_RW={p.get('lambda_RW', 1.0)}")
    print(f"  qiz_us_fixed_cost_mode={p.get('qiz_us_fixed_cost_mode', 'stacked')}")
    print(f"  mfn_us_fixed_cost_discount_sigma={p.get('mfn_us_fixed_cost_discount_sigma', {})}")
    print(f"  sigma_C={p['sigma_C']}")
    print("Key calibration:")
    print(f"  sigma={p['sigma']}")
    print(f"  theta={p['theta']}")
    print(f"  alpha={p['alpha']}")
    print(f"  delta={p['delta']}")
    print(f"  f_upgrade={p['f_upgrade']}")
    if p.get("upgrade_mode", "binary") == "continuous":
        print(f"  upgrade_psi={p['upgrade_psi']}")
        print(f"  upgrade_psi_comp={p['upgrade_psi_comp']}")
        print(f"  upgrade_cost_fixed={p['upgrade_cost_fixed']}")
        print(f"  upgrade_cost_lin={p['upgrade_cost_lin']}")
        print(f"  upgrade_cost_quad={p['upgrade_cost_quad']}")
        print(f"  upgrade_cost_comp_mult={p.get('upgrade_cost_comp_mult', {})}")
        print(f"  upgrade_intensity_max={p['upgrade_intensity_max']}")
        print(f"  upgrade_intensity_grid_size={p['upgrade_intensity_grid_size']}")
    print(f"  t_mfn={p['t_mfn']}")
    print(f"  gamma={p['gamma']}")
    print(f"  fC_mean={p['fC_mean']}")
    print(f"  f_dom={p['f_dom']}")
    print(f"  f_export={p['f_export']}")
    print(f"  f_entry={p['f_entry']}")
    print(f"  beta={p['beta']}")
    print(f"  E_foreign={p['E_foreign']}")
    print(f"  P_foreign={p['P_foreign']}")
    print(f"  T_transfer={p['T_transfer']}")
    print("Numerics (do not change economics):")
    print(f"  n_phi={p['n_phi']} n_eps={p['n_eps']} goods_tol={p['goods_tol']} outer_tol={p['outer_tol']}")
    print(f"  outer_step={p['outer_step']} outer_cycle_tol={p.get('outer_cycle_tol', None)}")
    print(f"  entry_mass_floor={p['entry_mass_floor']} entry_mass_corner_tol={p.get('entry_mass_corner_tol', None)}")


def build_params_from_preset(preset: str) -> Dict[str, Any]:
    if preset == "paper":
        return params_defensible()
    if preset == "expected_direction_base":
        return params_expected_direction_base()
    if preset == "institutional_transparent":
        return params_institutional_transparent()
    if preset == "interior_gamma":
        return params_interior_gamma()
    if preset == "data_like":
        return params_data_like()
    if preset == "upgrade_complementarity":
        return params_upgrade_complementarity()
    if preset == "fit_exports_dual_market":
        return params_fit_exports_dual_market()
    if preset == "fit_us_aggregate_positive":
        return params_fit_us_aggregate_positive()
    raise ValueError(f"Unknown preset: {preset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QIZ model runner")
    parser.add_argument(
        "--preset",
        choices=[
            "paper",
            "expected_direction_base",
            "institutional_transparent",
            "interior_gamma",
            "data_like",
            "upgrade_complementarity",
            "fit_exports_dual_market",
            "fit_us_aggregate_positive",
        ],
        default="paper",
        help="Calibration preset. Default is transparent paper baseline."
    )
    parser.add_argument(
        "--fit-stylized",
        action="store_true",
        help="Run transparent coarse-grid fit to declared stylized moments."
    )
    parser.add_argument(
        "--freeze-baseline",
        action="store_true",
        help="Solve QIZ on/off once and write a frozen baseline snapshot JSON."
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Compare current baseline run to a frozen baseline snapshot JSON."
    )
    parser.add_argument(
        "--baseline-file",
        default="baseline_freeze_fit_exports_dual_market.json",
        help="Path to baseline snapshot JSON used by --freeze-baseline / --check-baseline."
    )
    parser.add_argument(
        "--baseline-tol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for baseline moment drift checks."
    )
    args = parser.parse_args()

    p = build_params_from_preset(args.preset)
    print_assumption_report(p, args.preset)

    if args.freeze_baseline or args.check_baseline:
        print("\nRunning baseline snapshot solve (QIZ on/off)...")
        comp = compare_qiz_on_off(p, disable_upgrade=False)
        snapshot = baseline_snapshot_from_result(p, comp, args.preset)
        print_baseline_table(snapshot)

        if args.freeze_baseline:
            write_baseline_snapshot(args.baseline_file, snapshot)
            print(f"Baseline snapshot written to: {args.baseline_file}")

        if args.check_baseline:
            out = check_baseline_snapshot(args.baseline_file, snapshot, tol=args.baseline_tol)
            if not out["diffs"]:
                print(f"Baseline check passed: no moment drift above tol={args.baseline_tol:g}")
            else:
                print(f"Baseline check found {len(out['diffs'])} moment drifts above tol={args.baseline_tol:g}")
                for k, cur_v, old_v, d in out["diffs"][:25]:
                    print(f"  {k}: current={cur_v:.10g}, frozen={old_v:.10g}, abs_diff={d:.3e}")
        raise SystemExit(0)

    if args.fit_stylized:
        print("\nRunning transparent stylized-moment fit...")
        p_fit = copy.deepcopy(p)
        # Keep fitting time manageable; final validation can be rerun at finer grids.
        p_fit["n_phi"] = min(p_fit["n_phi"], 25)
        p_fit["n_eps"] = max(p_fit["n_eps"], 3)
        p_fit["outer_max_iter"] = max(p_fit["outer_max_iter"], 1500)
        p_fit["outer_tol"] = max(p_fit["outer_tol"], 2.0e-3)
        p_fit["outer_cycle_tol"] = max(p_fit.get("outer_cycle_tol", 2.0e-3), 8.0e-3)
        p_fit["upgrade_mode"] = "continuous"
        p_fit["roo_cost_scope"] = "US_only"

        targets, weights = default_tiered_moment_spec()
        grid = {
            "upgrade_psi_T": [0.10, 0.14],
            "upgrade_psi_comp_T": [0.06, 0.10],
            "upgrade_cost_quad_T": [4.0, 4.8],
            "t_mfn_T": [0.10, 0.15, 0.20],
            "fC_mean_T": [0.10, 0.20],
        }
        best = fit_to_stylized_moments(
            p_fit,
            targets,
            weights,
            grid,
            minimum_only=True,
            verbose=True
        )
        p = best["p"]
        fit_res = best["result"]
        base = fit_res["on"]
        off = fit_res["off"]
        tr_on = fit_res["trade_on"]
        tr_off = fit_res["trade_off"]

        print("\nFit targets:", targets)
        print("Fit weights:", weights)
        print("Best-fit knobs:", best["params"])
        print("Best-fit loss:", best["loss"])
        print("Best-fit moments:", best["moments"])
        print("Tiered target moments:", {
            k: best["moments"].get(k, np.nan) for k in targets
        })
    else:
        print("Solving baseline equilibrium (QIZ on)...")
        base = solve_equilibrium(p, qiz_on=True, disable_upgrade=False, verbose=True)
        try:
            off = solve_equilibrium(p, qiz_on=False, disable_upgrade=False, initial_state=base, verbose=False)
            tr_on = summarize_trade(base, p)
            tr_off = summarize_trade(off, p)
        except RuntimeError:
            # Fallback: solve the pair jointly (reliable warm starts and cycle handling).
            comp = compare_qiz_on_off(p, disable_upgrade=False)
            base = comp["on"]
            off = comp["off"]
            tr_on = comp["trade_on"]
            tr_off = comp["trade_off"]

    print("\nBaseline welfare:", base["welfare"])
    print("Baseline wages:", base["w"])
    print("Baseline employment shares:", {r: base["Ls"][r]/p["L_total"] for r in p["regions"]})
    print("Baseline sector prices:", base["goods"]["P_EG_s"])
    print("Baseline moments (Q,T):", base["moments"][("Q","T")])

    print("\nPolicy comparison: QIZ on vs QIZ off...")
    for j in ["US", "RW"]:
        key = ("Q", j)
        X_on = tr_on["exports"][key]
        X_off = tr_off["exports"][key]
        dX = X_on - X_off
        pct = 100.0 * dX / max(abs(X_off), 1e-12)
        print(f"Q exports to {j}: on={X_on:.6f}, off={X_off:.6f}, delta={dX:.6f} ({pct:.4f}%)")
        N_on = tr_on["exporter_masses"][key]
        N_off = tr_off["exporter_masses"][key]
        x_on = tr_on["avg_revenue_per_exporter"][key]
        x_off = tr_off["avg_revenue_per_exporter"][key]
        ext = (N_on - N_off) * x_off
        inten = N_off * (x_on - x_off)
        inter = (N_on - N_off) * (x_on - x_off)
        print(f"  decomposition: extensive={ext:.6f}, intensive={inten:.6f}, interaction={inter:.6f}")

    print("Real wages (w/P):")
    for r in p["regions"]:
        print(f"  {r} on={base['w'][r]/base['goods']['P_EG']:.6f} off={off['w'][r]/off['goods']['P_EG']:.6f}")
    print(f"Aggregate welfare (Y/P): on={base['welfare']:.6f} off={off['welfare']:.6f}")
    print(
        "Q,T group summary (on-state):",
        summarize_qt_groups(base, p, qiz_on=True, disable_upgrade=base.get("disable_upgrade", False))
    )
    print(
        "Q,T same-type on/off changes:",
        summarize_qt_same_type_changes(base, off, p, disable_upgrade=base.get("disable_upgrade", False))
    )

    print("\nCounterfactual A: Shut down productivity channel (delta=1)...")
    cfA = counterfactual_shutdown_productivity(p, baseline=base)
    print("Welfare baseline:", cfA["baseline"]["welfare"])
    print("Welfare no-productivity:", cfA["no_productivity"]["welfare"])
    print("Q,T moments baseline:", cfA["baseline"]["moments"][("Q","T")])
    print("Q,T moments no-productivity:", cfA["no_productivity"]["moments"][("Q","T")])

    print("\nCounterfactual B: Vary textiles ROO requirement gamma_T...")
    gamma_list = [0.0, 0.01, 0.05, 0.105, 0.15]
    path = counterfactual_gamma_path(p, sector="T", gamma_list=gamma_list, start_state=base)
    for row in path:
        print(row)
