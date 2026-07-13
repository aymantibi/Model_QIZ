"""
invert_trade_costs.py

Inverts bilateral import shares pi_ijs into iceberg trade costs d_ijs
for the three-region QIZ model: Egypt (EGY), US, RoW.

Inversion formula (standard EK/Melitz-Pareto)
----------------------------------------------
The Melitz-Pareto share equation is:

    pi_ijs = (A_is * d_ijs^{-theta_s}) / sum_k (A_ks * d_kjs^{-theta_s})

where A_is = T_is * w_is^{-theta_s} is the composite exporter fundamental
(Pareto scale T_is times wage w_is). Taking the ratio with the domestic share:

    pi_jjs / pi_ijs = (A_js / A_is) * d_ijs^{theta_s}    [d_jjs = 1]

=> d_ijs * (A_js/A_is)^{1/theta_s} = (pi_jjs / pi_ijs)^{1/theta_s}

Without separately identifying T_is and w_is, we cannot disentangle d_ijs
from A_is. The standard EK approach is to absorb both into a composite:

    tau_ijs = (A_js/A_is)^{1/theta_s} * d_ijs = (pi_jjs / pi_ijs)^{1/theta_s}

This is the bilateral trade wedge — it includes pure iceberg trade costs PLUS
relative exporter competitiveness. It is the standard object used in EK (2002)
and most Ricardian/Melitz-Pareto trade models for counterfactual analysis.

Note on wage correction
-----------------------
Separating tau into d (pure trade cost) and A (competitiveness) requires
wage data AND TFP data. With only wages, the correction overcorrects because
it attributes Egypt's wage advantage entirely to trade cost reduction rather
than TFP differences. Given that Egypt's wages are ~20x lower than the US
but Egypt is also less productive, the wage correction produces nonsensical
d < 1 values. We therefore use the composite tau_ijs throughout.

For counterfactual QIZ analysis, the QIZ shock is a tariff elimination:
tau_EGY_US_post = tau_EGY_US_pre * (1 + t_MFN)^{-1}
which operates on the trade cost component only. The A_is terms cancel in the
ratio, so the composite tau is the correct calibration target.

Steps
-----
1. Load bilateral flows X_ijs from trade_shares.json.
2. Load absorption E_js from data_calibration/absorption_2004.json.
3. Compute pi_ijs = X_ijs / E_js and domestic shares pi_jjs = DS_js / E_js.
4. Invert: tau_ijs = (pi_jjs / pi_ijs)^{1/theta_s}.
5. Save trade_costs.json.

Source for wages (saved for reference, not used in inversion):
  data_calibration/unido_output_wages_employment_2004.csv
  w_js = Wages_USD / Employees by ISIC sector-region, UNIDO 2004.
"""

import json
import pandas as pd

BASE = 'C:/Users/Admin/Desktop/Idea QIZs and Development/Model'
DATA = f'{BASE}/data_calibration'

# ------------------------------------------------------------------
# 1. Load inputs
# ------------------------------------------------------------------
with open(f'{BASE}/trade_shares.json') as f:
    ts = json.load(f)

with open(f'{DATA}/absorption_2004.json') as f:
    ab = json.load(f)

with open(f'{BASE}/params_estimated.json') as f:
    params = json.load(f)

# Load wages for reference/documentation only
uw = pd.read_csv(f'{DATA}/unido_output_wages_employment_2004.csv', encoding='latin1')

# ------------------------------------------------------------------
# 2. Sector mappings
# ------------------------------------------------------------------
def isic_to_sector_g1(isic):
    isic = int(isic)
    if isic in [17, 18]:       return 'T'
    elif isic in [15, 16, 23]: return 'NON_MFG'
    else:                      return 'O'

def isic_to_sector_g2(isic):
    isic = int(isic)
    if isic in [17, 18]:   return 'T'
    elif isic in [15, 16]: return 'S1'
    elif isic in [24, 25]: return 'S2'
    elif isic == 26:       return 'S3'
    elif isic == 23:       return 'NON_MFG'
    else:                  return 'O'

# ------------------------------------------------------------------
# 3. Compute wage per worker for reference
# ------------------------------------------------------------------
uw2 = uw[uw['Year'] == 2004].copy()
uw2['isic']    = pd.to_numeric(uw2['ActivityCode'], errors='coerce')
uw2['val_usd'] = pd.to_numeric(uw2['ValueUSD'],    errors='coerce')
uw2['val_n']   = pd.to_numeric(uw2['Value'],       errors='coerce')
uw2 = uw2.dropna(subset=['isic'])
uw2['isic'] = uw2['isic'].astype(int)
uw2 = uw2[uw2['isic'].between(15, 37)]
uw2['g2'] = uw2['isic'].apply(isic_to_sector_g2)

def region(name):
    name = str(name).strip()
    if name == 'Egypt':                    return 'EGY'
    if name == 'United States of America': return 'US'
    return 'RoW'

uw2['region'] = uw2['Country'].apply(region)

wages_g2 = uw2[uw2['Variable'] == 'Wages and salaries'].groupby(['region','g2'])['val_usd'].sum().unstack('region').fillna(0)
empl_g2  = uw2[uw2['Variable'] == 'Employees'].groupby(['region','g2'])['val_n'].sum().unstack('region').fillna(0)

print("=== Wage per worker w_js = W_js_USD / L_js (reference, not used in inversion) ===")
for s in sorted([s for s in wages_g2.index if s != 'NON_MFG']):
    def g(df, reg):
        try: return float(df.loc[s, reg])
        except: return 0.0
    W_EGY = g(wages_g2,'EGY'); L_EGY = g(empl_g2,'EGY')
    W_US  = g(wages_g2,'US');  L_US  = g(empl_g2,'US')
    W_RoW = g(wages_g2,'RoW'); L_RoW = g(empl_g2,'RoW')
    w_EGY = W_EGY/L_EGY if L_EGY>0 else None
    w_US  = W_US /L_US  if L_US >0 else None
    w_RoW = W_RoW/L_RoW if L_RoW>0 else None
    print(f"  {s}: w_EGY=${w_EGY:,.0f}  w_US=${w_US:,.0f}  w_RoW=${w_RoW:,.0f}  | w_US/w_EGY={w_US/w_EGY:.1f}x  w_RoW/w_EGY={w_RoW/w_EGY:.1f}x")

# ------------------------------------------------------------------
# 4. Helper functions
# ------------------------------------------------------------------
def safe_div(a, b):
    return a / b if b and b > 0 else None

def invert_tau(pi_jj, pi_ij, theta):
    """
    tau_ijs = (pi_jjs / pi_ijs)^{1/theta_s}
    Composite trade wedge = (A_js/A_is)^{1/theta} * d_ijs.
    """
    if pi_ij and pi_ij > 0 and pi_jj and pi_jj > 0:
        return round((pi_jj / pi_ij) ** (1.0 / theta), 6)
    return None

# ------------------------------------------------------------------
# 5. Build trade costs for each grouping
# ------------------------------------------------------------------
def build_costs(grp_key):
    flows_data = ts[grp_key]['data']
    abs_data   = ab[grp_key]
    theta_data = params['theta_s'][grp_key]['theta']

    results = {}
    for s in sorted(abs_data.keys()):
        ab_s  = abs_data[s]
        fld   = flows_data.get(s, {}).get('flows_usd', {})
        theta = theta_data[s]

        # Absorption expenditures
        E_EGY = ab_s['E_EGY']
        E_US  = ab_s['E_US']
        E_RoW = ab_s['E_RoW']

        # Domestic sales = output - total exports
        DS_EGY = ab_s['Y_EGY'] - ab_s['X_EGY']
        DS_US  = ab_s['Y_US']  - ab_s['X_US']
        DS_RoW = ab_s['Y_RoW'] - ab_s['X_RoW']

        # Bilateral flows from trade data
        X_EGY_US  = fld.get('X_EGY_US',  0.0)
        X_EGY_RoW = fld.get('X_EGY_RoW', 0.0)
        X_US_EGY  = fld.get('X_US_EGY',  0.0)
        X_US_RoW  = fld.get('X_US_RoW',  0.0)
        X_RoW_EGY = max(E_EGY - DS_EGY - X_US_EGY, 0.0)
        X_RoW_US  = max(E_US  - DS_US  - X_EGY_US, 0.0)

        # Import shares (absorption denominators)
        pi_EGY_US  = safe_div(X_EGY_US,  E_US)
        pi_RoW_US  = safe_div(X_RoW_US,  E_US)
        pi_US_US   = safe_div(DS_US,      E_US)

        pi_US_EGY  = safe_div(X_US_EGY,  E_EGY)
        pi_RoW_EGY = safe_div(X_RoW_EGY, E_EGY)
        pi_EGY_EGY = safe_div(DS_EGY,    E_EGY)

        pi_EGY_RoW = safe_div(X_EGY_RoW, E_RoW)
        pi_US_RoW  = safe_div(X_US_RoW,  E_RoW)
        pi_RoW_RoW = safe_div(DS_RoW,    E_RoW)

        def rnd(x): return round(x, 6) if x is not None else None

        # Composite trade wedges tau_ijs = (pi_jjs/pi_ijs)^{1/theta}
        tau_EGY_US  = invert_tau(pi_US_US,   pi_EGY_US,  theta)
        tau_RoW_US  = invert_tau(pi_US_US,   pi_RoW_US,  theta)
        tau_US_EGY  = invert_tau(pi_EGY_EGY, pi_US_EGY,  theta)
        tau_RoW_EGY = invert_tau(pi_EGY_EGY, pi_RoW_EGY, theta)
        tau_EGY_RoW = invert_tau(pi_RoW_RoW, pi_EGY_RoW, theta)
        tau_US_RoW  = invert_tau(pi_RoW_RoW, pi_US_RoW,  theta)

        results[s] = {
            'theta': theta,
            'expenditure_absorption_usd': {
                'E_EGY': round(E_EGY, 0),
                'E_US':  round(E_US,  0),
                'E_RoW': round(E_RoW, 0),
            },
            'domestic_sales_usd': {
                'DS_EGY': round(DS_EGY, 0),
                'DS_US':  round(DS_US,  0),
                'DS_RoW': round(DS_RoW, 0),
            },
            'import_shares_absorption': {
                'pi_EGY_US':  rnd(pi_EGY_US),  'pi_RoW_US':  rnd(pi_RoW_US),  'pi_US_US':   rnd(pi_US_US),
                'pi_US_EGY':  rnd(pi_US_EGY),  'pi_RoW_EGY': rnd(pi_RoW_EGY), 'pi_EGY_EGY': rnd(pi_EGY_EGY),
                'pi_EGY_RoW': rnd(pi_EGY_RoW), 'pi_US_RoW':  rnd(pi_US_RoW),  'pi_RoW_RoW': rnd(pi_RoW_RoW),
            },
            'trade_costs': {
                'note': 'tau_ijs = (pi_jjs/pi_ijs)^{1/theta_s}. Composite wedge: tau = (A_js/A_is)^{1/theta} * d_ijs. tau_jjs=1.',
                'tau_EGY_US':  tau_EGY_US,
                'tau_RoW_US':  tau_RoW_US,
                'tau_US_EGY':  tau_US_EGY,
                'tau_RoW_EGY': tau_RoW_EGY,
                'tau_EGY_RoW': tau_EGY_RoW,
                'tau_US_RoW':  tau_US_RoW,
                'tau_EGY_EGY': 1.0,
                'tau_US_US':   1.0,
                'tau_RoW_RoW': 1.0,
            }
        }

    return results

g1 = build_costs('grouping_1')
g2 = build_costs('grouping_2')

# ------------------------------------------------------------------
# 6. Save trade_costs.json
# ------------------------------------------------------------------
output = {
    'description': 'Bilateral trade wedges tau_ijs for QIZ model. Standard EK inversion from absorption-based import shares.',
    'year': 2004,
    'method': 'tau_ijs = (pi_jjs / pi_ijs)^{1/theta_s}',
    'interpretation': 'tau_ijs is the composite trade wedge = (A_js/A_is)^{1/theta} * d_ijs. Includes both pure iceberg trade costs and relative exporter competitiveness (TFP x wage). Cannot be decomposed without TFP data.',
    'normalization': 'tau_jjs = 1 for all j, s',
    'counterfactual_use': 'QIZ tariff shock: tau_EGY_US_post = tau_EGY_US_pre / (1 + t_MFN_s). A_is terms cancel in the share ratio under counterfactual so composite wedge is the correct object.',
    'theta_source': 'params_estimated.json: Hill+OLS from Egyptian customs panel',
    'E_js_source': 'data_calibration/absorption_2004.json: E_js = Y - X + M, UNIDO + Comtrade world 2004',
    'wage_reference': 'data_calibration/unido_output_wages_employment_2004.csv: w_js = W_USD/L by ISIC. Saved for reference. Not used in inversion — wage correction requires joint TFP identification.',
    'units': 'dimensionless',
    'grouping_1': {'sectors': ['T', 'O'], 'data': g1},
    'grouping_2': {'sectors': ['T', 'S1', 'S2', 'S3', 'O'], 'data': g2},
}

with open(f'{BASE}/trade_costs.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nSaved: trade_costs.json")
print()

# ------------------------------------------------------------------
# 7. Summary
# ------------------------------------------------------------------
for grp, label in [('grouping_1','Grouping 1'), ('grouping_2','Grouping 2')]:
    print(f"\n{'='*80}")
    print(f"  {label} — composite trade wedges tau_ijs, 2004")
    print(f"{'='*80}")
    for s, v in sorted(output[grp]['data'].items()):
        d  = v['trade_costs']
        th = v['theta']
        pi = v['import_shares_absorption']
        print(f"\n  Sector {s}  (theta={th})")
        print(f"  Domestic shares:  pi_EGY_EGY={pi['pi_EGY_EGY']:.4f}  pi_US_US={pi['pi_US_US']:.4f}  pi_RoW_RoW={pi['pi_RoW_RoW']:.4f}")
        print(f"  Wedges:  tau_EGY->US={d['tau_EGY_US']}  tau_RoW->US={d['tau_RoW_US']}  tau_US->EGY={d['tau_US_EGY']}  tau_RoW->EGY={d['tau_RoW_EGY']}  tau_EGY->RoW={d['tau_EGY_RoW']}  tau_US->RoW={d['tau_US_RoW']}")
