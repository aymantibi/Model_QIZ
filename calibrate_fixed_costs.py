#!/usr/bin/env python3
"""
calibrate_fixed_costs.py
------------------------
Joint calibration of all fixed costs in the QIZ model (grouping_1: T, O).

Calibration sequence
--------------------
Stage 1 — f_dom, f_export (joint, nested Brent)
    Target moments (Egypt Enterprise Survey 2004, unweighted):
        dom_only_T  = 0.8927   share of T manufacturers serving domestic only
        dom_only_O  = 0.8492   share of O manufacturers serving domestic only
        exp_US_T    = 0.0575   share of T manufacturers exporting to US
        exp_US_O    = 0.0112   share of O manufacturers exporting to US
        exp_RW_T    = 0.0690   share of T manufacturers exporting to RW
        exp_RW_O    = 0.1425   share of O manufacturers exporting to RW

    Parameters calibrated (one per moment, 6 total):
        f_dom[s]             same for Q and N (no region-level participation data)
        f_export[(r,US,s)]   same for Q and N
        f_export[(r,RW,s)]   same for Q and N

    Strategy: outer loop over f_dom (2 params), inner Brent loops over
    f_export_US and f_export_RW for each sector. Iterate until all 6
    moments match simultaneously.

Stage 2 — f_entry (Brent per sector)
    Target moments (Industrial Stats 2004):
        QIZ firm share_T = n_QT / (n_QT + n_NT) = 1013 / 1140 = 0.8886
        QIZ firm share_O = n_QO / (n_QO + n_NO) = 6218 / 7708 = 0.8067
    Normalize f_entry[(N,s)] = 1 for each sector.
    Calibrate f_entry[(Q,s)] to match active firm share M_Qs*E_active / total.

Stage 3 — fC_mean[T] (Brent)
    Target: compliance_rate = 0.321
    Source: customs panel 2005-2016

Stage 4 — f_upgrade[T] (Brent)
    Target: upgrading_rate = 0.491
    Source: customs panel 2005-2016

All results saved to params_estimated.json under 'calibrated_fixed_costs'.
"""

import json
import os
import copy
import numpy as np
from scipy.optimize import brentq
from qiz_model_ge import params_defensible, solve_equilibrium

OUT_PATH = os.path.join(os.path.dirname(__file__), "params_estimated.json")

# ─────────────────────────────────────────────────────────────────────────────
# Speed: use reduced n_phi grid during calibration (Brent search).
# Full-resolution run is done only for the final saved equilibria.
# ─────────────────────────────────────────────────────────────────────────────
N_PHI_CALIB = 60    # fast Brent iterations
N_PHI_FINAL = 140   # final equilibria

def fast_p(p):
    """Return a copy of p with reduced n_phi for fast calibration solves."""
    p2 = copy.deepcopy(p)
    p2["n_phi"] = N_PHI_CALIB
    return p2

# ─────────────────────────────────────────────────────────────────────────────
# Target moments
# ─────────────────────────────────────────────────────────────────────────────

TARGETS = {
    # Stage 1: participation rates (ECES 2004, grouping_1)
    "dom_only": {"T": 0.8927, "O": 0.8492},
    "exp_US":   {"T": 0.0575, "O": 0.0112},
    "exp_RW":   {"T": 0.0690, "O": 0.1425},
    # Stage 2: QIZ firm shares (Industrial Stats 2004, grouping_1)
    "qiz_firm_share": {"T": 1013 / (1013 + 127), "O": 6218 / (6218 + 1490)},
    # Stage 3: compliance rate (customs panel 2005-2016)
    "compliance_rate": 0.321,
    # Stage 4: upgrading rate (customs panel 2005-2016)
    "upgrading_rate": 0.491,
}

SECTORS  = ["T", "O"]
REGIONS  = ["Q", "N"]
TOLS     = {"stage1": 5e-4, "stage2": 5e-4, "stage3": 1e-4, "stage4": 1e-4}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_fdom(p, fdom_T, fdom_O):
    p = copy.deepcopy(p)
    for r in REGIONS:
        p["f_dom"][(r, "T")] = fdom_T
        p["f_dom"][(r, "O")] = fdom_O
    return p

def set_fexport(p, fUS_T, fRW_T, fUS_O, fRW_O):
    p = copy.deepcopy(p)
    for r in REGIONS:
        p["f_export"][(r, "US", "T")] = fUS_T
        p["f_export"][(r, "RW", "T")] = fRW_T
        p["f_export"][(r, "US", "O")] = fUS_O
        p["f_export"][(r, "RW", "O")] = fRW_O
    return p

def set_fentry(p, fE_Q_T, fE_Q_O):
    p = copy.deepcopy(p)
    p["f_entry"][("Q", "T")] = fE_Q_T
    p["f_entry"][("Q", "O")] = fE_Q_O
    p["f_entry"][("N", "T")] = 1.0
    p["f_entry"][("N", "O")] = 1.0
    return p

def solve(p, qiz_on=True, warm=None):
    return solve_equilibrium(
        p, qiz_on=qiz_on,
        initial_state=warm,
        verbose=False
    )

def get_participation(sol, s):
    """
    Aggregate participation moments across Q and N using active-firm masses.

    The survey targets are shares among operating manufacturers, not labor-share
    weighted shares and not shares among entrants. We therefore aggregate using
    M_rs * E_active_rs as the denominator and direct cache indicators for
    domestic-only and export participation.
    """
    M = sol["M"]
    cache = sol["goods"]["cache"]
    active_mass = dom_mass = us_mass = rw_mass = any_export_mass = 0.0
    for r in REGIONS:
        m = cache[(r, s)]
        region_scale = M[(r, s)]
        active_mass += region_scale * m["E_active"]
        dom_mass += region_scale * m["E_dom_only"]
        us_mass += region_scale * m["E_US"]
        rw_mass += region_scale * m["E_RW"]
        any_export_mass += region_scale * m["E_any_export"]

    denom = max(active_mass, 1e-12)
    return {
        "dom_only": dom_mass / denom,
        "exp_US": us_mass / denom,
        "exp_RW": rw_mass / denom,
        "any_export": any_export_mass / denom,
        "active_mass": active_mass,
    }

def choose_highest_cost_hitting_target(
    obj, target, lo=0.01, hi=25.0, tol=1e-3, maxiter=30, max_hi=500.0
):
    """
    For a weakly decreasing share function obj(cost), choose the highest cost
    whose implied share is still weakly above the target.

    This is more robust than exact root-finding on the finite productivity grid,
    where participation moments are step functions.
    """
    val_lo = obj(lo)
    val_hi = obj(hi)
    if val_lo < target:
        raise RuntimeError(
            f"Target {target:.4f} is unattainable even at lowest cost {lo:.4f} "
            f"(share={val_lo:.4f})."
        )

    while (val_hi >= target) and (hi < max_hi):
        hi = min(max_hi, 2.0 * hi)
        val_hi = obj(hi)
    if val_hi >= target:
        return hi, val_hi

    best_cost, best_val = lo, val_lo
    left, right = lo, hi
    for _ in range(maxiter):
        mid = 0.5 * (left + right)
        val_mid = obj(mid)
        if val_mid >= target:
            best_cost, best_val = mid, val_mid
            left = mid
        else:
            right = mid
        if right - left < tol:
            break
    return best_cost, best_val

def choose_cost_closest_to_target(
    obj, target, lo=0.01, hi=5.0, tol=1e-4, maxiter=30, max_hi=500.0
):
    """
    For a weakly decreasing moment function obj(cost), return the cost whose
    predicted moment is closest to target. Works even when the function is a
    step function on the finite productivity grid.
    """
    val_lo = obj(lo)
    val_hi = obj(hi)
    while (val_hi > target) and (hi < max_hi):
        hi = min(max_hi, 2.0 * hi)
        val_hi = obj(hi)

    if (val_lo - target) * (val_hi - target) > 0:
        return min([(lo, val_lo), (hi, val_hi)], key=lambda kv: abs(kv[1] - target))

    left, right = lo, hi
    y_left, y_right = val_lo, val_hi
    for _ in range(maxiter):
        mid = 0.5 * (left + right)
        y_mid = obj(mid)
        if abs(y_mid - target) < tol:
            left = right = mid
            y_left = y_right = y_mid
            break
        if (y_left - target) * (y_mid - target) <= 0:
            right, y_right = mid, y_mid
        else:
            left, y_left = mid, y_mid
        if right - left < tol:
            break

    cand = [(left, y_left), ((left + right) * 0.5, obj((left + right) * 0.5)), (right, y_right)]
    return min(cand, key=lambda kv: abs(kv[1] - target))

def get_qiz_firm_share(sol, s):
    """M_Qs * E_active_Qs / (M_Qs*E_active_Qs + M_Ns*E_active_Ns)."""
    M = sol["M"]
    cache = sol["goods"]["cache"]
    nQ = M[("Q", s)] * cache[("Q", s)]["E_active"]
    nN = M[("N", s)] * cache[("N", s)]["E_active"]
    return nQ / max(nQ + nN, 1e-12)

def get_compliance_rate(sol):
    """Compliance uptake among Q,T US exporters."""
    return sol["moments"][("Q", "T")]["compliance_share_among_US_exporters"]

def get_upgrading_rate(sol):
    """upgrade_share_among_active for (Q, T)."""
    return sol["moments"][("Q", "T")]["upgrade_share_among_active"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: joint calibration of f_dom and f_export
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_stage1(p_base, verbose=True):
    """
    Jointly calibrate f_dom[s] and f_export[(r,j,s)] to participation targets.

    Strategy: for given f_dom values, use Brent to find f_export_US and
    f_export_RW per sector that match exp_US and exp_RW targets. Then use
    an outer fixed-point loop over f_dom until dom_only targets also match.
    """
    if verbose:
        print("\n" + "="*65)
        print("  STAGE 1: Calibrating f_dom and f_export")
        print("="*65)

    def inner_brent_sector(p, s, fdom_s, verbose_inner=False):
        """
        For fixed f_dom[s], find f_export_US[s] and f_export_RW[s]
        that weakly hit exp_US and exp_RW targets for sector s.
        We choose the highest fixed cost that still keeps exporter shares
        at or above target, which respects the user's tolerance for shares
        slightly above the observed moments and avoids fragile exact roots
        on a lumpy finite grid.
        """
        def rw_share(fRW):
            p2 = fast_p(p)
            for r in REGIONS:
                p2["f_dom"][(r, s)] = fdom_s
                p2["f_export"][(r, "RW", s)] = fRW
            try:
                sol = solve(p2, qiz_on=False)
            except Exception as exc:
                raise RuntimeError(
                    f"solve() failed in rw_share: fRW={fRW:.6f}, sector={s}, "
                    f"fdom_s={fdom_s:.4f}"
                ) from exc
            part = get_participation(sol, s)
            if verbose_inner:
                print(f"      fRW[{s}]={fRW:.4f} -> exp_RW={part['exp_RW']:.4f} (target={TARGETS['exp_RW'][s]:.4f})")
            return part["exp_RW"]

        fRW_star, rw_star = choose_highest_cost_hitting_target(
            rw_share, TARGETS["exp_RW"][s], tol=TOLS["stage1"], maxiter=30
        )

        def us_share(fUS):
            p2 = fast_p(p)
            for r in REGIONS:
                p2["f_dom"][(r, s)] = fdom_s
                p2["f_export"][(r, "RW", s)] = fRW_star
                p2["f_export"][(r, "US", s)] = fUS
            try:
                sol = solve(p2, qiz_on=False)
            except Exception as exc:
                raise RuntimeError(
                    f"solve() failed in us_share: fUS={fUS:.6f}, sector={s}"
                ) from exc
            part = get_participation(sol, s)
            if verbose_inner:
                print(f"      fUS[{s}]={fUS:.4f} -> exp_US={part['exp_US']:.4f} (target={TARGETS['exp_US'][s]:.4f})")
            return part["exp_US"]

        fUS_star, us_star = choose_highest_cost_hitting_target(
            us_share, TARGETS["exp_US"][s], tol=TOLS["stage1"], maxiter=30
        )
        if verbose_inner:
            print(
                f"      chosen exporter costs for {s}: "
                f"fUS={fUS_star:.4f} -> {us_star:.4f}, "
                f"fRW={fRW_star:.4f} -> {rw_star:.4f}"
            )
        return fUS_star, fRW_star

    # Outer loop over f_dom: fixed-point iteration
    # Initial guesses
    fdom = {"T": 1.0, "O": 1.0}
    fUS  = {"T": 2.0, "O": 5.0}
    fRW  = {"T": 1.5, "O": 2.0}

    MAX_OUTER = 12
    best_state = None
    best_max_err = np.inf
    for it in range(MAX_OUTER):
        p_try = copy.deepcopy(p_base)
        for s in SECTORS:
            for r in REGIONS:
                p_try["f_dom"][(r, s)]        = fdom[s]
                p_try["f_export"][(r, "US", s)] = fUS[s]
                p_try["f_export"][(r, "RW", s)] = fRW[s]

        # Inner: for each sector find f_export given current f_dom
        for s in SECTORS:
            if verbose:
                print(f"\n  Outer it={it+1}, sector={s}: f_dom={fdom[s]:.4f} -> finding f_export...")
            fUS[s], fRW[s] = inner_brent_sector(p_try, s, fdom[s], verbose_inner=verbose)
            if verbose:
                print(f"    f_export_US[{s}]={fUS[s]:.4f}  f_export_RW[{s}]={fRW[s]:.4f}")

        # Check dom_only moments with updated f_export
        p_check = copy.deepcopy(p_base)
        for s in SECTORS:
            for r in REGIONS:
                p_check["f_dom"][(r, s)]          = fdom[s]
                p_check["f_export"][(r, "US", s)]  = fUS[s]
                p_check["f_export"][(r, "RW", s)]  = fRW[s]

        sol_check = solve(fast_p(p_check), qiz_on=False)
        errors_dom = {}
        for s in SECTORS:
            part = get_participation(sol_check, s)
            errors_dom[s] = part["dom_only"] - TARGETS["dom_only"][s]
            if verbose:
                print(f"  Sector {s}: dom_only={part['dom_only']:.4f} "
                      f"(target={TARGETS['dom_only'][s]:.4f}, err={errors_dom[s]:+.4f}) "
                      f"exp_US={part['exp_US']:.4f} exp_RW={part['exp_RW']:.4f}")

        max_err = max(abs(e) for e in errors_dom.values())
        if max_err < best_max_err:
            best_max_err = max_err
            best_state = (
                copy.deepcopy(fdom),
                copy.deepcopy(fUS),
                copy.deepcopy(fRW),
                sol_check,
            )
        if max_err < TOLS["stage1"] * 2:
            if verbose:
                print(f"\n  Stage 1 converged (max dom_only error={max_err:.4f})")
            break

        # Update f_dom: if dom_only too high (too many domestic-only firms),
        # lower f_dom to make domestic market less attractive relative to exporting.
        # dom_only too high -> domestic profit relatively too easy -> raise f_dom.
        for s in SECTORS:
            # Gradient: higher f_dom -> fewer serve domestic -> lower dom_only
            # So to reduce dom_only error (dom too high), increase f_dom
            fdom[s] = max(0.01, fdom[s] * (1.0 + 0.3 * errors_dom[s]))

    if best_state is not None:
        best_fdom, best_fUS, best_fRW, best_sol = best_state
        if verbose and best_max_err >= TOLS["stage1"] * 2:
            print(f"\n  Stage 1 stopped at iteration cap; returning best fit with max dom_only error={best_max_err:.4f}")
        return best_fdom, best_fUS, best_fRW, best_sol

    return fdom, fUS, fRW, sol_check


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: f_entry
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_stage2(p_base, verbose=True):
    """
    Calibrate f_entry[(Q,s)] per sector to match QIZ active firm share.
    f_entry[(N,s)] = 1.0 (normalization).
    """
    if verbose:
        print("\n" + "="*65)
        print("  STAGE 2: Calibrating f_entry")
        print("="*65)

    results = {}
    p_cur = copy.deepcopy(p_base)

    for s in SECTORS:
        target = TARGETS["qiz_firm_share"][s]
        last_sol = [None]

        def pred_share(fE_Q):
            p2 = fast_p(p_cur)
            p2["f_entry"][("Q", s)] = fE_Q
            p2["f_entry"][("N", s)] = 1.0
            try:
                sol = solve(p2, qiz_on=False, warm=last_sol[0])
            except RuntimeError:
                sol = solve(p2, qiz_on=False, warm=None)
            last_sol[0] = sol
            pred = get_qiz_firm_share(sol, s)
            if verbose:
                print(f"    f_entry_Q[{s}]={fE_Q:.4f} -> qiz_share={pred:.4f} (target={target:.4f})")
            return pred

        grid = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20, 1.60, 2.20, 3.00, 4.00]
        grid_vals = []
        for x in grid:
            try:
                grid_vals.append((x, pred_share(x)))
            except RuntimeError:
                continue

        if not grid_vals:
            raise RuntimeError(f"Stage 2 could not evaluate any stable grid points for sector {s}.")

        fE_star, best_y = min(grid_vals, key=lambda kv: abs(kv[1] - target))
        if verbose:
            print(
                f"  Closest stable stage-2 point for sector {s}: "
                f"f_entry_Q={fE_star:.4f} with qiz_share={best_y:.4f}."
            )

        results[s] = fE_star
        p_cur["f_entry"][("Q", s)] = fE_star
        p_cur["f_entry"][("N", s)] = 1.0
        if verbose:
            print(f"  f_entry[(Q,{s})] = {fE_star:.4f}  f_entry[(N,{s})] = 1.0")

    return results, p_cur


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: fC_mean[T]
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_stage3(p_base, verbose=True):
    if verbose:
        print("\n" + "="*65)
        print("  STAGE 3: Calibrating fC_mean[T]")
        print("="*65)

    target = TARGETS["compliance_rate"]

    last_sol = [None]

    def pred_rate(fC):
        p2 = fast_p(p_base)
        p2["fC_mean"]["T"] = fC
        try:
            sol = solve(p2, qiz_on=True, warm=last_sol[0])
        except RuntimeError:
            sol = solve(p2, qiz_on=True, warm=None)
        last_sol[0] = sol
        pred = get_compliance_rate(sol)
        if verbose:
            print(f"    fC[T]={fC:.4f} -> compliance={pred:.4f} (target={target:.4f})")
        return pred

    grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.30, 2.00, 3.00, 5.00]
    vals = []
    for x in grid:
        try:
            vals.append((x, pred_rate(x)))
        except RuntimeError:
            continue
    if not vals:
        raise RuntimeError("Stage 3 could not find any stable candidate.")
    fC_star, pred_star = min(vals, key=lambda kv: abs(kv[1] - target))
    if verbose:
        print(f"  fC_mean[T] = {fC_star:.4f} -> compliance={pred_star:.4f}")
    return fC_star


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: f_upgrade[T]
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_stage4(p_base, verbose=True):
    if verbose:
        print("\n" + "="*65)
        print("  STAGE 4: Calibrating f_upgrade[T]")
        print("="*65)

    target = TARGETS["upgrading_rate"]

    last_sol = [None]

    def pred_rate(fU):
        p2 = fast_p(p_base)
        p2["f_upgrade"]["T"] = fU
        try:
            sol = solve(p2, qiz_on=True, warm=last_sol[0])
        except RuntimeError:
            sol = solve(p2, qiz_on=True, warm=None)
        last_sol[0] = sol
        pred = get_upgrading_rate(sol)
        if verbose:
            print(f"    f_upgrade[T]={fU:.4f} -> upgrading={pred:.4f} (target={target:.4f})")
        return pred

    grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.30, 2.00, 3.00, 5.00]
    vals = []
    for x in grid:
        try:
            vals.append((x, pred_rate(x)))
        except RuntimeError:
            continue
    if not vals:
        raise RuntimeError("Stage 4 could not find any stable candidate.")
    fU_star, pred_star = min(vals, key=lambda kv: abs(kv[1] - target))
    if verbose:
        print(f"  f_upgrade[T] = {fU_star:.4f} -> upgrading={pred_star:.4f}")
    return fU_star


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    p = params_defensible()

    print("=" * 65)
    print("  QIZ MODEL — FULL FIXED COST CALIBRATION")
    print("  Grouping 1: T (textiles), O (other mfg)")
    print("=" * 65)
    print("  Targets:")
    for k, v in TARGETS.items():
        print(f"    {k}: {v}")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    fdom, fUS, fRW, sol1 = calibrate_stage1(p, verbose=True)

    for s in SECTORS:
        for r in REGIONS:
            p["f_dom"][(r, s)]          = fdom[s]
            p["f_export"][(r, "US", s)] = fUS[s]
            p["f_export"][(r, "RW", s)] = fRW[s]

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    fentry_Q, p = calibrate_stage2(p, verbose=True)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    fC_star = calibrate_stage3(p, verbose=True)
    if fC_star is not None:
        p["fC_mean"]["T"] = fC_star

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    fU_star = calibrate_stage4(p, verbose=True)
    if fU_star is not None:
        p["f_upgrade"]["T"] = fU_star

    # ── Final equilibrium: QIZ on vs off ─────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL EQUILIBRIUM: QIZ ON vs QIZ OFF")
    print("="*65)

    sol_on  = solve(p, qiz_on=True)
    sol_off = solve(p, qiz_on=False)

    print("\n  QIZ ON:")
    print(f"    Welfare:  {sol_on['welfare']:.6f}")
    print(f"    w_Q:      {sol_on['w']['Q']:.4f}   w_N: {sol_on['w']['N']:.4f}")
    for s in SECTORS:
        for r in REGIONS:
            m = sol_on["moments"][(r, s)]
            print(f"    ({r},{s}): active={m['active_share']:.3f}  US={m['US_export_share_among_active']:.3f}  "
                  f"comp={m['compliance_share_among_active']:.3f}  up={m['upgrade_share_among_active']:.3f}")

    print("\n  QIZ OFF:")
    print(f"    Welfare:  {sol_off['welfare']:.6f}")
    print(f"    w_Q:      {sol_off['w']['Q']:.4f}   w_N: {sol_off['w']['N']:.4f}")
    for s in SECTORS:
        for r in REGIONS:
            m = sol_off["moments"][(r, s)]
            print(f"    ({r},{s}): active={m['active_share']:.3f}  US={m['US_export_share_among_active']:.3f}  "
                  f"comp={m['compliance_share_among_active']:.3f}  up={m['upgrade_share_among_active']:.3f}")

    print("\n  QIZ effect (on - off):")
    print(f"    Welfare gain: {sol_on['welfare'] - sol_off['welfare']:+.6f} "
          f"({(sol_on['welfare']/sol_off['welfare']-1)*100:+.3f}%)")
    print(f"    w_Q change:   {(sol_on['w']['Q']/sol_off['w']['Q']-1)*100:+.3f}%")
    print(f"    w_N change:   {(sol_on['w']['N']/sol_off['w']['N']-1)*100:+.3f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(OUT_PATH) as f:
        params_out = json.load(f)

    params_out["calibrated_fixed_costs"] = {
        "description": (
            "All fixed costs calibrated jointly to match observed participation "
            "and selection moments. Grouping 1 (T, O). QIZ on for stages 2-4."
        ),
        "method": "Sequential Brent root-finding with joint outer loop for stages 1-2.",
        "grouping_1": {
            "f_dom":    {s: round(fdom[s], 4) for s in SECTORS},
            "f_export_US": {s: round(fUS[s], 4) for s in SECTORS},
            "f_export_RW": {s: round(fRW[s], 4) for s in SECTORS},
            "f_entry_Q":   {s: round(fentry_Q[s], 4) for s in SECTORS},
            "f_entry_N":   {s: 1.0 for s in SECTORS},
            "fC_mean_T":   round(fC_star, 4) if fC_star else None,
            "f_upgrade_T": round(fU_star, 4) if fU_star else None,
        },
        "targets": TARGETS,
        "qiz_on_welfare":  round(sol_on["welfare"],  6),
        "qiz_off_welfare": round(sol_off["welfare"], 6),
        "welfare_gain_pct": round((sol_on["welfare"] / sol_off["welfare"] - 1) * 100, 4),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(params_out, f, indent=2)

    print(f"\n  Saved to: {OUT_PATH}")
