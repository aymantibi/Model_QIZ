/*****************************************************************
*  estimate_kappa.do
*
*  Estimates the labor mobility elasticity kappa for the two-region
*  QIZ/non-QIZ model from ELMPS panel data.
*
*  Spec A: Governorate-level migration-share OLS
*     log(M_ogt / M_oot) = alpha_og + kappa*log(w_gt) + lambda_t
*     FE: od-pair + interval | Cluster: od-pair
*
*  QIZ governorate codes (ELMPS gov variable):
*    Treated by 2004/2005: 1,2,3,4,11,12,13,14,16,17,19,21
*    Treated by 2009:      + 22,24
*    Treated by 2013:      + 15,18
*  QIZ status is time-varying: a gov is QIZ only from its treatment year onward.
*
*  Source for gov codes: ATT Staggered diff in diff.do (treatedgov variable)
*
*  Output: kappa_estimate.csv
*****************************************************************/

cd "C:\Users\Admin\Desktop\Idea QIZs and Development\Model"

use "C:\Users\Admin\Desktop\Idea QIZs and Development\ELMPS 2018 Everythings\All Data\Up to 2023\elmps 2023 panel 98_23 v2.0.dta", clear

*==================================================================
* STEP 1: RESHAPE TO LONG AND CONSTRUCT KEY VARIABLES
*==================================================================

cap keep Findid gov_* RhrwgCPI23_* usemp1_* age_* ///
     panel_wt_98_06_12_18_23 panel_wt_98_06_12_18 panel_wt_98_06_12 ///
     panel_wt_98_06 panel_wt_06_12 panel_wt_12_18

* Rename wave suffixes to 4-digit years
foreach prefix in gov_ RhrwgCPI23_ usemp1_ age_ {
    cap rename `prefix'98   `prefix'1998
    cap rename `prefix'06   `prefix'2006
    cap rename `prefix'12   `prefix'2012
    cap rename `prefix'18   `prefix'2018
    cap rename `prefix'23   `prefix'2023
}

* Collapse panel weights into a single variable before reshape
gen panel_wt = panel_wt_98_06_12_18_23
replace panel_wt = panel_wt_98_06_12_18 if missing(panel_wt)
replace panel_wt = panel_wt_98_06_12    if missing(panel_wt)
replace panel_wt = panel_wt_98_06       if missing(panel_wt)
replace panel_wt = panel_wt_06_12       if missing(panel_wt)
replace panel_wt = panel_wt_12_18       if missing(panel_wt)

* Reshape to long
reshape long gov_ RhrwgCPI23_ usemp1_ age_, i(Findid) j(year)

rename gov_         gov
rename RhrwgCPI23_  realwage
rename usemp1_      employed
rename age_         age

keep if inlist(year, 1998, 2006, 2012, 2018, 2023)

* Log real wage (employed workers only)
gen log_realwage = log(realwage) if employed == 1 & realwage > 0 & !missing(realwage)

sort Findid year

*==================================================================
* STEP 2: GOVERNORATE-LEVEL WAGE MEASURES
*==================================================================

preserve
    keep if employed == 1 & !missing(log_realwage)
    collapse (mean) w_gov = log_realwage (count) n_emp = log_realwage, ///
        by(gov year)
    save "temp_gov_wages.dta", replace
restore

*==================================================================
* STEP 3: BILATERAL FLOW COUNTS (WORKER LEVEL → COLLAPSED)
*==================================================================

* Reshape wide for transition analysis
keep Findid year gov employed panel_wt
reshape wide gov employed, i(Findid) j(year)

* Define transitions
local start_yrs 1998 2006 2012 2018
local end_yrs   2006 2012 2018 2023

* QIZ set is time-varying — define per transition end year:
*   By 2006 (treated 2004/2005): 1,2,3,4,11,12,13,14,16,17,19,21
*   By 2012 (+ treated 2009):    + 22,24
*   By 2018 (+ treated 2013):    + 15,18
*   By 2023: same as 2018

forvalues k = 1/4 {
    local t1 : word `k' of `start_yrs'
    local t2 : word `k' of `end_yrs'

    preserve
        keep if !missing(gov`t1') & !missing(gov`t2')
        keep Findid gov`t1' gov`t2'
        rename gov`t1' gov_o
        rename gov`t2' gov_d
        gen interval = `t1'
        gen one = 1
        collapse (sum) n_flows = one, by(gov_o gov_d interval)
        save "temp_flows_`t1'_`t2'.dta", replace
    restore
}

use "temp_flows_1998_2006.dta", clear
append using "temp_flows_2006_2012.dta"
append using "temp_flows_2012_2018.dta"
append using "temp_flows_2018_2023.dta"

* Expand to full gov_o x gov_d x interval grid (including zero-flow pairs for PPML)
fillin gov_o gov_d interval
replace n_flows = 0 if missing(n_flows)

* Time-varying QIZ indicator for destination gov
gen qiz_dest = 0
replace qiz_dest = inlist(gov_d, 1,2,3,4,11,12,13,14,16,17,19,21) ///
    if interval == 1998
replace qiz_dest = inlist(gov_d, 1,2,3,4,11,12,13,14,16,17,19,21,22,24) ///
    if interval == 2006
replace qiz_dest = inlist(gov_d, 1,2,3,4,11,12,13,14,15,16,17,18,19,21,22,24) ///
    if interval == 2012
replace qiz_dest = inlist(gov_d, 1,2,3,4,11,12,13,14,15,16,17,18,19,21,22,24) ///
    if interval == 2018

* Merge destination wages (end-of-interval wave)
gen year_dest = .
replace year_dest = 2006 if interval == 1998
replace year_dest = 2012 if interval == 2006
replace year_dest = 2018 if interval == 2012
replace year_dest = 2023 if interval == 2018

rename gov_d gov
rename year_dest year
merge m:1 gov year using "temp_gov_wages.dta", keepusing(w_gov) keep(1 3) nogen
rename w_gov  log_wage_dest
rename gov    gov_d
rename year   year_dest
drop year_dest

* OD-pair fixed effect
egen od_pair = group(gov_o gov_d)

*==================================================================
* SPEC A: PPML — od-pair FE + interval FE, cluster at od-pair
*==================================================================

di _newline "========================================================"
di "  SPEC A: Governorate migration-share PPML"
di "  DV: n_flows (levels) | RHS: log(w_dest)"
di "  FE: od_pair + interval | Cluster: od_pair"
di "========================================================"

* --- PPML baseline: od-pair FE + interval FE ---
di _newline "  PPML: n_flows ~ exp(kappa*log(w_dest) + od_FE + interval_FE)"
ppmlhdfe n_flows log_wage_dest if gov_o != gov_d, ///
    absorb(od_pair interval) cluster(od_pair)
local kappa_ppml    = _b[log_wage_dest]
local kappa_ppml_se = _se[log_wage_dest]
di "  kappa (PPML) = " %6.4f `kappa_ppml' "  SE = " %6.4f `kappa_ppml_se'

* --- PPML robustness: origin x interval FE ---
di _newline "  PPML (rich FE): n_flows ~ exp(kappa*log(w_dest) + od_FE + gov_o#interval_FE)"
egen gov_o_X_interval = group(gov_o interval)
ppmlhdfe n_flows log_wage_dest if gov_o != gov_d, ///
    absorb(od_pair gov_o_X_interval) cluster(od_pair)
local kappa_ppml_rich    = _b[log_wage_dest]
local kappa_ppml_rich_se = _se[log_wage_dest]
di "  kappa (PPML, rich FE) = " %6.4f `kappa_ppml_rich' "  SE = " %6.4f `kappa_ppml_rich_se'

*==================================================================
* SAVE RESULTS
*==================================================================

clear
set obs 2
gen str50  spec        = ""
gen str200 description = ""
gen double kappa       = .
gen double se          = .
gen str200 notes       = ""

replace spec        = "A_PPML" in 1
replace description = "PPML: n_flows ~ exp(kappa*log(w_dest) + od_FE + interval_FE)" in 1
replace kappa       = `kappa_ppml'    in 1
replace se          = `kappa_ppml_se' in 1
replace notes       = "Retains zero-flow od-pairs; cluster: od_pair; QIZ time-varying" in 1

replace spec        = "A_PPML_richFE" in 2
replace description = "PPML: n_flows ~ exp(kappa*log(w_dest) + od_FE + gov_o#interval_FE)" in 2
replace kappa       = `kappa_ppml_rich'    in 2
replace se          = `kappa_ppml_rich_se' in 2
replace notes       = "Absorbs origin-specific time-varying push shocks; cluster: od_pair" in 2

export delimited using "kappa_estimate.csv", replace
di _newline "  Saved: kappa_estimate.csv"

* Cleanup temp files
erase "temp_gov_wages.dta"
erase "temp_flows_1998_2006.dta"
erase "temp_flows_2006_2012.dta"
erase "temp_flows_2012_2018.dta"
erase "temp_flows_2018_2023.dta"
