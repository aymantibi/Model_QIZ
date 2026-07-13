/*****************************************************************
*  compliance_and_upgraders.do
*
*  Computes two target moments for model calibration:
*
*  1. COMPLIANCE RATE
*     Among textile firms that ever export to the US (full sample),
*     the share that ever import from Israel (full sample).
*
*  2. UPGRADING RATE
*     Among textile firms that ever import from Israel (full sample),
*     the share whose non-US exports increase after their first
*     Israeli import relative to the year before.
*
*  Sample: is_textile == 1, Years 2005-2016
*
*  Output: compliance_and_upgraders.csv
*****************************************************************/

cd "C:\Users\Admin\Desktop\Idea QIZs and Development\Model"

use "C:\Users\Admin\Desktop\Idea QIZs and Development\Export and Import Data Egypt\final_data_matched.dta", clear

keep if is_textile == 1
keep if Year >= 2005 & Year <= 2016

* Firm-level ever indicators
gen exports_us  = (exp_from_us    > 0) & !missing(exp_from_us)
gen imports_isr = (imp_from_israel > 0) & !missing(imp_from_israel)

bysort Trader_ID: egen ever_us  = max(exports_us)
bysort Trader_ID: egen ever_isr = max(imports_isr)


*==================================================================
* SECTION 1 — COMPLIANCE RATE
*
*  Denominator: firms that ever export to US (2005-2016)
*  Numerator:   firms that ever export to US AND ever import
*               from Israel (2005-2016)
*==================================================================

preserve
collapse (max) ever_us ever_isr, by(Trader_ID)

quietly count if ever_us == 1
local n_us = r(N)
quietly count if ever_us == 1 & ever_isr == 1
local n_comp = r(N)
local comp_rate = `n_comp' / `n_us'

di _newline "========================================================"
di "  Compliance Rate — Textile Firms, 2005-2016 (Full Sample)"
di "  Complier = ever imports from Israel"
di "  Denominator = ever exports to US"
di "========================================================"
di "  Ever US-exporting textile firms:  `n_us'"
di "  Of which ever import from Israel: `n_comp'"
di "  Compliance rate:                  " %6.4f `comp_rate'
di "========================================================"
restore


*==================================================================
* SECTION 2 — UPGRADING RATE
*
*  Denominator: firms that ever import from Israel (2005-2016)
*               with a valid treatment_year and both pre/post
*               non-US export observations
*  Numerator:   of those, firms with exp_from_nonus(t+1) >
*               exp_from_nonus(t-1), where t = treatment_year
*==================================================================

* Keep compliant firms with valid treatment year
keep if ever_isr == 1
keep if !missing(treatment_year)

keep Trader_ID treatment_year

* Merge back with full panel to get non-US exports at t-1 and t+1
joinby Trader_ID using "C:\Users\Admin\Desktop\Idea QIZs and Development\Export and Import Data Egypt\final_data_matched.dta", unmatched(master)

gen nonus_pre  = exp_from_nonus if Year == treatment_year - 1
gen nonus_post = exp_from_nonus if Year == treatment_year + 1

bysort Trader_ID: egen nonus_pre_val  = max(nonus_pre)
bysort Trader_ID: egen nonus_post_val = max(nonus_post)

collapse (first) treatment_year nonus_pre_val nonus_post_val, by(Trader_ID)

* Require both pre and post observed
keep if !missing(nonus_pre_val) & !missing(nonus_post_val)

gen upgrader = (nonus_post_val > nonus_pre_val)

quietly count
local n_obs = r(N)
quietly count if upgrader == 1
local n_up = r(N)
local up_rate = `n_up' / `n_obs'

di _newline "========================================================"
di "  Upgrading Rate — Compliant Textile Firms, 2005-2016"
di "  Complier  = ever imports from Israel"
di "  Upgrader  = exp_from_nonus(t+1) > exp_from_nonus(t-1)"
di "  where t   = first Israeli import year (treatment_year)"
di "========================================================"
di "  Compliant firms with pre+post observed: `n_obs'"
di "  Of which non-US exports increased:      `n_up'"
di "  Upgrading rate:                         " %6.4f `up_rate'
di "========================================================"


*==================================================================
* SAVE OUTPUT
*==================================================================

clear
set obs 2
gen str20  moment  = ""
gen int    n_denom = .
gen int    n_num   = .
gen double rate    = .

replace moment  = "compliance_rate" in 1
replace moment  = "upgrading_rate"  in 2
replace n_denom = `n_us'  in 1
replace n_denom = `n_obs' in 2
replace n_num   = `n_comp' in 1
replace n_num   = `n_up'   in 2
replace rate    = `comp_rate' in 1
replace rate    = `up_rate'   in 2

export delimited moment n_denom n_num rate ///
    using "compliance_and_upgraders.csv", replace

di _newline "  Saved: compliance_and_upgraders.csv"
