"""
build_trade_shares.py

Constructs bilateral trade flows X_ijs, expenditures E_js, and import
shares pi_ijs for the three-region QIZ model: Egypt (EGY), US, RoW.

Regions:  i,j in {EGY, US, RoW}
Sectors:  Grouping 1: T, O
          Grouping 2: T, S1, S2, S3, O

Sector HS chapter mappings:
  T  : HS 50-63  (Textiles + Apparel)
  S1 : HS 02,04,07-24  (Food products)
  S2 : HS 28-38  (Chemicals)
  S3 : HS 25,26,68-70  (Non-metallic minerals)
  O  : remaining manufacturing (excl. energy HS27, agriculture HS01-24)

Source: TradeData_4_16_2026_11_24_27.csv (new Comtrade download, 2003-2004)
Year:   2004 (pre-QIZ baseline)

Column mapping (shifted by 1 due to leading 'C' field):
  isOriginalClassification -> HS4 product code
  fobvalue                 -> trade value in USD
  refPeriodId              -> year (2003 or 2004)
  reporterCode             -> reporter ISO (EGY, USA)
  partnerCode              -> partner (USA, W00=World, EGY)
  flowCode                 -> Import or Export

Flow construction:
  X_EGY_US  : USA reporter, Import from EGY
  X_EGY_RoW : EGY reporter, Export to W00 minus Export to USA
  X_US_EGY  : EGY reporter, Import from USA
  X_US_RoW  : USA reporter, Export to W00 minus Export to EGY
  X_RoW_EGY : EGY reporter, Import from W00 minus Import from USA
  X_RoW_US  : USA reporter, Import from W00 minus Import from EGY

Expenditures:
  E_EGY = EGY reporter, Import from W00  (total Egypt imports)
  E_US  = USA reporter, Import from W00  (total US imports)
  E_RoW = X_EGY_RoW + X_US_RoW          (RoW expenditure on EGY and US goods)

Import shares:
  pi_ijs = X_ijs / E_js

Output: trade_shares.json
"""

import pandas as pd
import json

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(
    'C:/Users/Admin/Desktop/Idea QIZs and Development/Model/TradeData_4_16_2026_11_24_27.csv',
    encoding='latin1'
)

# Keep 2004 only
df = df[df['refPeriodId'] == 2004].copy()

# Rename shifted columns to correct names
df['hs4']      = pd.to_numeric(df['isOriginalClassification'], errors='coerce')
df['value']    = pd.to_numeric(df['fobvalue'], errors='coerce')
df['year']     = df['refPeriodId']
df['reporter'] = df['reporterCode']
df['partner']  = df['partnerCode']
df['flow']     = df['flowCode']

df = df.dropna(subset=['hs4', 'value'])
df['chapter'] = df['hs4'].apply(lambda c: int(str(int(c)).zfill(4)[:2]))

# ------------------------------------------------------------------
# 2. Sector classifications
# ------------------------------------------------------------------
def sector_g1(ch):
    if 50 <= ch <= 63:               return 'T'
    elif ch <= 24 or ch == 27:       return 'NON_MFG'
    else:                            return 'O'

def sector_g2(ch):
    if 50 <= ch <= 63:               return 'T'
    elif ch in [2,4] or 7<=ch<=24:   return 'S1'
    elif 28 <= ch <= 38:             return 'S2'
    elif ch in [25,26] or 68<=ch<=70: return 'S3'
    elif ch <= 24 or ch == 27:       return 'NON_MFG'
    else:                            return 'O'

df['g1'] = df['chapter'].apply(sector_g1)
df['g2'] = df['chapter'].apply(sector_g2)

# ------------------------------------------------------------------
# 3. Helper: sum value for a given reporter/partner/flow/sector
# ------------------------------------------------------------------
def get_flow(reporter, partner, flow, sector_col):
    mask = (
        (df['reporter'] == reporter) &
        (df['partner']  == partner)  &
        (df['flow']     == flow)
    )
    return df[mask].groupby(sector_col)['value'].sum()

# ------------------------------------------------------------------
# 4. Build bilateral flow matrix and compute E_js, pi_ijs
# ------------------------------------------------------------------
def build_matrix(scol):
    # Raw aggregates by sector
    # Into US (USA reporter)
    usa_imp_egy = get_flow('USA', 'EGY', 'Import', scol)  # X_EGY->US
    usa_imp_wld = get_flow('USA', 'W00', 'Import', scol)  # E_US (total US imports)
    usa_exp_egy = get_flow('USA', 'EGY', 'Export', scol)  # X_US->EGY (mirror, not used)
    usa_exp_wld = get_flow('USA', 'W00', 'Export', scol)  # US total exports to world

    # Into EGY (EGY reporter)
    egy_imp_usa = get_flow('EGY', 'USA', 'Import', scol)  # X_US->EGY
    egy_imp_wld = get_flow('EGY', 'W00', 'Import', scol)  # E_EGY (total Egypt imports)
    egy_exp_usa = get_flow('EGY', 'USA', 'Export', scol)  # Egypt exports to US
    egy_exp_wld = get_flow('EGY', 'W00', 'Export', scol)  # Egypt total exports to world

    sectors = sorted(set(
        list(usa_imp_wld.index) + list(egy_imp_wld.index)
    ))
    sectors = [s for s in sectors if s != 'NON_MFG']

    def g(series, s): return series.get(s, 0.0)

    results = {}
    for s in sectors:
        # --- Expenditures ---
        E_US  = g(usa_imp_wld, s)   # total US imports from world
        E_EGY = g(egy_imp_wld, s)   # total Egypt imports from world

        # --- Flows into US ---
        X_EGY_US = g(usa_imp_egy, s)              # USA reporter: imports from EGY
        X_RoW_US = max(E_US - X_EGY_US, 0)        # residual

        # --- Flows into EGY ---
        X_US_EGY  = g(egy_imp_usa, s)             # EGY reporter: imports from USA
        X_RoW_EGY = max(E_EGY - X_US_EGY, 0)     # residual

        # --- Flows into RoW ---
        X_EGY_RoW = max(g(egy_exp_wld,s) - g(egy_exp_usa,s), 0)  # EGY reporter
        X_US_RoW  = max(g(usa_exp_wld,s) - g(usa_exp_egy,s), 0)  # USA reporter
        E_RoW     = X_EGY_RoW + X_US_RoW          # RoW expenditure on EGY+US goods

        # --- Import shares pi_ijs = X_ijs / E_js ---
        def share(num, den): return round(num/den, 6) if den > 0 else None

        results[s] = {
            'flows_usd': {
                'X_EGY_US':  round(X_EGY_US,  0),
                'X_EGY_RoW': round(X_EGY_RoW, 0),
                'X_US_EGY':  round(X_US_EGY,  0),
                'X_US_RoW':  round(X_US_RoW,  0),
                'X_RoW_EGY': round(X_RoW_EGY, 0),
                'X_RoW_US':  round(X_RoW_US,  0),
            },
            'expenditure_usd': {
                'E_EGY': round(E_EGY, 0),
                'E_US':  round(E_US,  0),
                'E_RoW': round(E_RoW, 0),
            },
            'import_shares': {
                'pi_EGY_US':  share(X_EGY_US,  E_US),
                'pi_RoW_US':  share(X_RoW_US,  E_US),
                'pi_US_EGY':  share(X_US_EGY,  E_EGY),
                'pi_RoW_EGY': share(X_RoW_EGY, E_EGY),
                'pi_EGY_RoW': share(X_EGY_RoW, E_RoW),
                'pi_US_RoW':  share(X_US_RoW,  E_RoW),
            }
        }

    return results

# ------------------------------------------------------------------
# 5. Compute for both groupings
# ------------------------------------------------------------------
g1 = build_matrix('g1')
g2 = build_matrix('g2')

# ------------------------------------------------------------------
# 6. Save to JSON
# ------------------------------------------------------------------
output = {
    'description': 'Bilateral trade flows, expenditures, and import shares for QIZ model. Regions: EGY, US, RoW.',
    'source': 'UN Comtrade, TradeData_4_16_2026_11_24_27.csv. Reporters: EGY and USA.',
    'year': 2004,
    'units': 'flows and expenditures in USD; shares dimensionless',
    'notes': {
        'E_EGY': 'Total Egypt imports from World (EGY reporter, W00 partner, Import)',
        'E_US':  'Total US imports from World (USA reporter, W00 partner, Import)',
        'E_RoW': 'RoW expenditure on EGY+US goods = X_EGY_RoW + X_US_RoW',
        'pi_ijs': 'X_ijs / E_js. Used to recover d_ijs via model elasticity.',
        'P_js':  'Not estimated. Solved inside model in equilibrium.',
    },
    'grouping_1': {
        'sectors': ['T', 'O'],
        'definitions': {
            'T': 'Textiles + Apparel (HS 50-63)',
            'O': 'Other manufacturing (excl. T, energy, agriculture)',
        },
        'data': g1
    },
    'grouping_2': {
        'sectors': ['T', 'S1', 'S2', 'S3', 'O'],
        'definitions': {
            'T':  'Textiles + Apparel (HS 50-63)',
            'S1': 'Food products (HS 02,04,07-24)',
            'S2': 'Chemicals (HS 28-38)',
            'S3': 'Non-metallic minerals (HS 25,26,68-70)',
            'O':  'Other manufacturing (residual)',
        },
        'data': g2
    }
}

with open('C:/Users/Admin/Desktop/Idea QIZs and Development/Model/trade_shares.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Saved: trade_shares.json")
print()

# ------------------------------------------------------------------
# 7. Print summary
# ------------------------------------------------------------------
for grp, label in [('grouping_1','Grouping 1'), ('grouping_2','Grouping 2')]:
    print(f"\n{'='*65}")
    print(f"  {label} — 2004")
    print(f"{'='*65}")
    for s, v in sorted(output[grp]['data'].items()):
        e = v['expenditure_usd']
        f = v['flows_usd']
        p = v['import_shares']
        print(f"\n  Sector {s}")
        print(f"  Expenditure (M USD):  E_EGY={e['E_EGY']/1e6:>8,.1f}  E_US={e['E_US']/1e6:>10,.1f}  E_RoW={e['E_RoW']/1e6:>10,.1f}")
        print(f"  Flows (M USD):  X_EGY->US={f['X_EGY_US']/1e6:>7,.1f}  X_EGY->RoW={f['X_EGY_RoW']/1e6:>7,.1f}  X_US->EGY={f['X_US_EGY']/1e6:>7,.1f}  X_US->RoW={f['X_US_RoW']/1e6:>9,.1f}  X_RoW->EGY={f['X_RoW_EGY']/1e6:>7,.1f}  X_RoW->US={f['X_RoW_US']/1e6:>9,.1f}")
        print(f"  Shares:  pi_EGY->US={p['pi_EGY_US']:.4f}  pi_RoW->US={p['pi_RoW_US']:.4f}  pi_US->EGY={p['pi_US_EGY']:.4f}  pi_RoW->EGY={p['pi_RoW_EGY']:.4f}  pi_EGY->RoW={p['pi_EGY_RoW']:.4f}  pi_US->RoW={p['pi_US_RoW']:.4f}")
