/*****************************************************************
*  estimate_entry_costs.do
*
*  Extracts firm count shares from the Egyptian Industrial Census
*  (2013 wave only) to calibrate f^E_rs (entry fixed costs).
*
*  The key moments are:
*    share_s_Q  = (sector-s firms in QIZ govs)    / (all mfg firms in QIZ govs)
*    share_s_N  = (sector-s firms in non-QIZ govs) / (all mfg firms in non-QIZ govs)
*    ratio_Q_to_N = share_s_Q / share_s_N
*
*  Sample restrictions:
*    - Private registered manufacturing firms: sector==2, tab16_9!=2, ecoact1d==3
*    - Exclude self-employed
*    - Exclude previously treated govs: 3,16,12,17,11,21,19,4,22,24,1,14,2,13
*    - QIZ govs: 15 and 18
*
*  Output: entry_costs_moments.csv          (grouping 1: T vs O)
*           entry_costs_moments_5sector.csv  (grouping 2: T/S1/S2/S3/O)
*****************************************************************/

cd "C:\Users\Admin\Desktop\Idea QIZs and Development\Model"

use "C:\Users\Admin\Desktop\Idea QIZs and Development\Firm Census\Egypt-Economic_Census-2013 (V1).dta", clear

* --- Sample restrictions ---

* Manufacturing firms only
keep if ecoact1d == 3

* Private registered firms only
keep if sector == 2
keep if tab16_9 != 2

* Drop self-employed
drop if selfEmployed == 1

* --- Generate indicators ---

* Grouping 1: T vs O
gen textile = inlist(ecoact2d, 13, 14)

* Grouping 2: T / S1 / S2 / S3 / O  (mirrors params_estimated.json sector_labels)
*   T  = Textiles + Wearing apparel  (ecoact2d 13, 14)
*   S1 = Food products               (ecoact2d 10)
*   S2 = Chemicals                   (ecoact2d 20, 21)
*   S3 = Non-metallic minerals       (ecoact2d 23)
*   O  = Other manufacturing (residual)
gen sector_g2 = "O"
replace sector_g2 = "T"  if inlist(ecoact2d, 13, 14)
replace sector_g2 = "S1" if inlist(ecoact2d, 10)
replace sector_g2 = "S2" if inlist(ecoact2d, 20, 21)
replace sector_g2 = "S3" if inlist(ecoact2d, 23)

gen qiz_gov = inlist(gov, 15, 18)

* Count establishments (each row is one firm)
gen one = 1

* ── PANEL A: grouping 1 (T vs O) × QIZ region ───────────────────────────────

preserve

collapse (sum) n_firms = one, by(qiz_gov textile)

bysort qiz_gov: egen n_total = sum(n_firms)
gen share = n_firms / n_total

sort qiz_gov textile
di _newline "========================================================"
di "  Grouping 1: Textile firm shares by QIZ region"
di "  (private registered manufacturing, 2013 census)"
di "========================================================"
list, sep(4)

* Reshape wide on qiz_gov, keep only textile rows for ratio
reshape wide n_firms n_total share, i(textile) j(qiz_gov)

gen ratio_Q_to_N = share1 / share0

keep if textile == 1

di _newline "========================================================"
di "  Ratio: textile share (QIZ) / textile share (non-QIZ)"
di "========================================================"
list share0 share1 ratio_Q_to_N

export delimited share0 share1 ratio_Q_to_N n_total0 n_total1 ///
    using "entry_costs_moments.csv", replace

di _newline "  Saved: entry_costs_moments.csv"

restore

* ── PANEL B: full 2-digit sector cross-tab for inspection ────────────────────

preserve

collapse (sum) n_firms = one, by(qiz_gov ecoact2d)

bysort qiz_gov: egen n_total = sum(n_firms)
gen share = n_firms / n_total

sort qiz_gov ecoact2d
di _newline "========================================================"
di "  Full 2-digit sector breakdown by region"
di "========================================================"
list, sep(0)

restore

* ── PANEL C: grouping 2 (T/S1/S2/S3/O) × QIZ region ────────────────────────

preserve

collapse (sum) n_firms = one, by(qiz_gov sector_g2)

bysort qiz_gov: egen n_total = sum(n_firms)
gen share = n_firms / n_total

sort qiz_gov sector_g2
di _newline "========================================================"
di "  Grouping 2 (T/S1/S2/S3/O) by region"
di "========================================================"
list, sep(0)

* Reshape wide on qiz_gov for ratio (j must be numeric)
reshape wide n_firms n_total share, i(sector_g2) j(qiz_gov)
gen ratio_Q_to_N = share1 / share0

sort sector_g2
di _newline "========================================================"
di "  Ratio: sector share (QIZ) / sector share (non-QIZ)"
di "========================================================"
list sector_g2 share0 share1 ratio_Q_to_N, sep(0)

export delimited sector_g2 share0 share1 ratio_Q_to_N n_total0 n_total1 ///
    using "entry_costs_moments_5sector.csv", replace

di _newline "  Saved: entry_costs_moments_5sector.csv"

restore
