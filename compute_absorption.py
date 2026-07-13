"""
compute_absorption.py

Computes sectoral absorption E_js = Y_js - X_js + M_js for regions j in {EGY, US, RoW}
and sectors s in both groupings.

RoW is constructed as: World aggregate - US - Egypt.

Sources:
  Output (Y):  data_calibration/unido_output_2004.csv  (UNIDO INDSTAT, ISIC Rev.3, 2004)
  Trade (X,M): data_calibration/comtrade_world_2004.csv (UN Comtrade, HS 2-digit, 2004)

ISIC -> sector mapping:
  T  : ISIC 17,18 (Textiles, Wearing apparel)
  S1 : ISIC 15,16 (Food, Beverages, Tobacco)
  S2 : ISIC 24,25 (Chemicals, Rubber/plastics)
  S3 : ISIC 26    (Non-metallic mineral products)
  O  : remaining manufacturing ISIC 20-37 excl. T,S1,S2,S3 and excl. ISIC 23 (petroleum)

HS chapter -> sector mapping (for trade):
  T  : HS 50-63
  S1 : HS 02,04,07-24
  S2 : HS 28-38
  S3 : HS 25,26,68-70
  O  : remaining manufacturing (excl. energy HS 27, raw agriculture HS 01)

Output: data_calibration/absorption_2004.json
"""

import pandas as pd
import json

BASE      = 'C:/Users/Admin/Desktop/Idea QIZs and Development/Model'
DATA      = f'{BASE}/data_calibration'
OUT_FILE  = f'{DATA}/absorption_2004.json'

# -----------------------------------------------------------------------
# 1. Load UNIDO output data
# -----------------------------------------------------------------------
unido = pd.read_csv(f'{DATA}/unido_output_2004.csv', encoding='latin1')

# Identify the relevant columns
# Expected: country code/name, ISIC code, year, output value
# Inspect column names
print("UNIDO columns:", list(unido.columns[:15]))
print(unido.head(3))

# -----------------------------------------------------------------------
# 2. Load world trade data (HS 2-digit chapters)
# -----------------------------------------------------------------------
trade = pd.read_csv(f'{DATA}/comtrade_world_2004.csv', encoding='latin1')
print("\nTrade columns:", list(trade.columns[:15]))
print(trade.head(3))

# -----------------------------------------------------------------------
# 3. Normalize column names based on actual file structure
# -----------------------------------------------------------------------

# --- UNIDO ---
# Rename to standardized names (adjust if column names differ)
unido_cols = {c.lower().strip(): c for c in unido.columns}

# Common UNIDO column patterns
def find_col(df, candidates):
    cols = {c.lower().strip() for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            # return original name
            for orig in df.columns:
                if orig.lower().strip() == c.lower():
                    return orig
    return None

isic_col    = find_col(unido, ['activitycode', 'isic3', 'isic', 'isic_3digit', 'isic3digit', 'ind_code', 'indcode'])
country_col = find_col(unido, ['country', 'countrycode', 'country_code', 'iso3', 'cty_code', 'reporter'])
year_col    = find_col(unido, ['year', 'ref_year', 'refyear'])
output_col  = find_col(unido, ['valueusd', 'output', 'gross_output', 'value_output', 'vos', 'out_usd', 'output_usd',
                                'indprod', 'ind_prod', 'production'])

print(f"\nUNIDO key columns found: isic={isic_col}, country={country_col}, year={year_col}, output={output_col}")

# --- Trade ---
# Comtrade world file: HS chapter is direct integer in isOriginalClassification
# but this file may have different structure — check
trade_cols = {c.lower().strip(): c for c in trade.columns}

rep_col     = find_col(trade, ['reportercode', 'reporter_code', 'reporter'])
partner_col = find_col(trade, ['partnercode', 'partner_code', 'partner'])
flow_col    = find_col(trade, ['flowcode', 'flow_code', 'flow'])
hs_col      = find_col(trade, ['isoriginalclassification', 'cmdcode', 'hs', 'hs_code', 'commodity'])
val_col     = find_col(trade, ['fobvalue', 'tradevalue', 'value', 'trade_value'])
year_t_col  = find_col(trade, ['refperiodid', 'year', 'ref_year', 'period'])

print(f"Trade key columns found: reporter={rep_col}, partner={partner_col}, flow={flow_col}, hs={hs_col}, value={val_col}, year={year_t_col}")

# -----------------------------------------------------------------------
# 4. Sector mappings
# -----------------------------------------------------------------------
def hs_to_sector_g1(ch):
    if 50 <= ch <= 63:                          return 'T'
    elif ch in [1] or (2 <= ch <= 24) or ch==27: return 'NON_MFG'
    else:                                        return 'O'

def hs_to_sector_g2(ch):
    if 50 <= ch <= 63:                           return 'T'
    elif ch in [2,4] or 7 <= ch <= 24:           return 'S1'
    elif 28 <= ch <= 38:                         return 'S2'
    elif ch in [25,26] or 68 <= ch <= 70:        return 'S3'
    elif ch in [1,3,5,6] or (7<=ch<=24) or ch==27: return 'NON_MFG'
    else:                                        return 'O'

def isic_to_sector_g1(isic):
    isic = int(isic)
    if isic in [17, 18]:                         return 'T'
    elif isic in [15, 16, 23]:                   return 'NON_MFG'
    else:                                        return 'O'

def isic_to_sector_g2(isic):
    isic = int(isic)
    if isic in [17, 18]:                         return 'T'
    elif isic in [15, 16]:                       return 'S1'
    elif isic in [24, 25]:                       return 'S2'
    elif isic == 26:                             return 'S3'
    elif isic == 23:                             return 'NON_MFG'
    else:                                        return 'O'

# -----------------------------------------------------------------------
# 5. Process UNIDO output
# -----------------------------------------------------------------------
u = unido.copy()
u = u[pd.to_numeric(u[year_col], errors='coerce') == 2004].copy()
u['isic_n'] = pd.to_numeric(u[isic_col], errors='coerce')
u['value_n'] = pd.to_numeric(u[output_col], errors='coerce')
u = u.dropna(subset=['isic_n', 'value_n'])
u['isic_n'] = u['isic_n'].astype(int)

# Keep only manufacturing ISIC 15-37
u = u[u['isic_n'].between(15, 37)]

u['g1'] = u['isic_n'].apply(isic_to_sector_g1)
u['g2'] = u['isic_n'].apply(isic_to_sector_g2)

# Country groupings
# EGY = Egypt, US = USA, RoW = everyone else
# Check what country codes look like
print("\nSample country codes in UNIDO:", u[country_col].unique()[:20])

egy_codes = ['EGY', 'Egypt', '818', 818]
usa_codes = ['USA', 'United States of America', '840', 840]

# UNIDO uses CountryCode (numeric) — also check Country name
def region(row):
    code = str(row[country_col]).strip()
    name = str(row['Country']).strip() if 'Country' in row.index else ''
    if code in [str(x) for x in egy_codes] or name in ['Egypt']: return 'EGY'
    if code in [str(x) for x in usa_codes] or name in ['United States of America', 'USA']: return 'US'
    return 'RoW'

u['region'] = u.apply(region, axis=1)

# Aggregate output by region x sector
def agg_output(grp_col):
    return u.groupby(['region', grp_col])['value_n'].sum().unstack('region').fillna(0)

Y_g1 = agg_output('g1')
Y_g2 = agg_output('g2')
print("\nUNIDO output by region (G1, USD):")
print(Y_g1)

# -----------------------------------------------------------------------
# 6. Process trade data
# -----------------------------------------------------------------------
tr = trade.copy()

# Keep 2004
if year_t_col:
    tr = tr[pd.to_numeric(tr[year_t_col], errors='coerce') == 2004].copy()

tr['hs_n'] = pd.to_numeric(tr[hs_col], errors='coerce')
tr['val_n'] = pd.to_numeric(tr[val_col], errors='coerce')
tr = tr.dropna(subset=['hs_n', 'val_n'])
tr['hs_n'] = tr['hs_n'].astype(int)

# This file uses HS chapter directly (1-99)
tr['g1'] = tr['hs_n'].apply(hs_to_sector_g1)
tr['g2'] = tr['hs_n'].apply(hs_to_sector_g2)

# Region of reporter
def region_trade(c):
    c = str(c).strip()
    if c in ['818', 'EGY', 'Egypt']: return 'EGY'
    if c in ['842', '840', 'USA', 'United States of America']: return 'US'
    return 'RoW'

tr['reporter_region'] = tr[rep_col].apply(region_trade)
tr['partner_region']  = tr[partner_col].apply(region_trade) if partner_col else 'unknown'

print("\nFlow codes in trade data:", tr[flow_col].unique() if flow_col else "N/A")
print("Reporter codes sample:", tr[rep_col].unique()[:10] if rep_col else "N/A")

# For each reporter: total exports to world and total imports from world
# We need:
#   X_j = total exports FROM j TO world
#   M_j = total imports INTO j FROM world
# Partner 'W00' or '0' = World aggregate

world_codes = ['W00', '0', 0, 'World', 'WLD']

def is_world(c):
    return str(c).strip() in [str(x) for x in world_codes]

tr['partner_is_world'] = tr[partner_col].apply(is_world) if partner_col else False

# Exports: reporter=j, flow=Export, partner=World
# Imports: reporter=j, flow=Import, partner=World
export_mask = tr['partner_is_world'] & (tr[flow_col].str.strip() == 'Export')
import_mask = tr['partner_is_world'] & (tr[flow_col].str.strip() == 'Import')

def agg_trade(mask, grp_col):
    return tr[mask].groupby(['reporter_region', grp_col])['val_n'].sum().unstack('reporter_region').fillna(0)

X_g1 = agg_trade(export_mask, 'g1')
M_g1 = agg_trade(import_mask, 'g1')
X_g2 = agg_trade(export_mask, 'g2')
M_g2 = agg_trade(import_mask, 'g2')

print("\nExports by reporter region (G1):")
print(X_g1)
print("\nImports by reporter region (G1):")
print(M_g1)

# -----------------------------------------------------------------------
# 7. Compute absorption E_js = Y_js - X_js + M_js
# -----------------------------------------------------------------------
def compute_absorption(Y, X, M, grouping_label):
    sectors = [s for s in Y.index if s != 'NON_MFG']
    results = {}
    for s in sectors:
        def g(df, region):
            try: return float(df.loc[s, region])
            except: return 0.0

        Y_EGY = g(Y, 'EGY'); X_EGY = g(X, 'EGY'); M_EGY = g(M, 'EGY')
        Y_US  = g(Y, 'US');  X_US  = g(X, 'US');  M_US  = g(M, 'US')
        Y_RoW = g(Y, 'RoW'); X_RoW = g(X, 'RoW'); M_RoW = g(M, 'RoW')

        results[s] = {
            'Y_EGY': Y_EGY, 'X_EGY': X_EGY, 'M_EGY': M_EGY,
            'E_EGY': max(Y_EGY - X_EGY + M_EGY, 0),
            'Y_US':  Y_US,  'X_US':  X_US,  'M_US':  M_US,
            'E_US':  max(Y_US  - X_US  + M_US,  0),
            'Y_RoW': Y_RoW, 'X_RoW': X_RoW, 'M_RoW': M_RoW,
            'E_RoW': max(Y_RoW - X_RoW + M_RoW, 0),
        }
        print(f"  {grouping_label} {s}: E_EGY={results[s]['E_EGY']/1e9:.2f}B  E_US={results[s]['E_US']/1e9:.2f}B  E_RoW={results[s]['E_RoW']/1e9:.2f}B")
    return results

print("\n--- Absorption (USD) ---")
g1_abs = compute_absorption(Y_g1, X_g1, M_g1, 'G1')
g2_abs = compute_absorption(Y_g2, X_g2, M_g2, 'G2')

# -----------------------------------------------------------------------
# 8. Save
# -----------------------------------------------------------------------
output = {'grouping_1': g1_abs, 'grouping_2': g2_abs}
with open(OUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT_FILE}")
