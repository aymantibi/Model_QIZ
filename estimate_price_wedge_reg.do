/*
estimate_price_wedge_reg.do
---------------------------
Estimates the Israeli input price premium relative to all other origins.

Spec 1 [preferred]:  log(uv_fhot) = alpha_fht + beta*1[ISR] + eps
  FE: firm x product x year
  Interpretation: same firm, same HS6, same year — how much more expensive
  are Israeli inputs relative to all other origins?

Spec 2 [robustness]: log(uv_fhot) = alpha_f + alpha_ht + beta*1[ISR] + eps
  FE: firm + product x year

exp(beta) = p_ISR / p_nonISR.  beta<0 => Israeli inputs cheaper.

Sample: manufacturing imports, all origins, 2005-2008.
SE clustered at HS6 x year.
*/

clear all
set more off

local data_dir `"C:\Users\Admin\Desktop\Idea QIZs and Development\Export and Import Data Egypt"'
local out_dir  `"C:\Users\Admin\Desktop\Idea QIZs and Development\Model"'

cap which reghdfe
if _rc {
    ssc install ftools,  replace
    ssc install reghdfe, replace
}

* ── load 2005-2008 ───────────────────────────────────────────────────────────
tempfile stacked
local first 1
foreach yr in 2005 2006 2007 2008 {
    local folder `"`data_dir'\EID-Imports-`yr' STATA"'
    local flist : dir `"`folder'"' files "*.dta", respectcase
    local fname = subinstr(`"`flist'"', `"""', "", .)
    local fname = trim(`"`fname'"')
    qui use `"`folder'\\`fname'"', clear
    if `first' {
        save `stacked', replace
        local first 0
    }
    else {
        append using `stacked'
        save `stacked', replace
    }
    di "Loaded `yr': `=_N' rows total"
}

* ── clean sample ─────────────────────────────────────────────────────────────
gen int hs2 = int(real(Product_HS6) / 10000)
keep if hs2 >= 2 & hs2 <= 97 & hs2 != 27   // manufacturing, drop petroleum
keep if Quantity > 0 & ImpVal_USD > 0

gen double log_uv = log(ImpVal_USD / Quantity)
keep if log_uv < .

* winsorise at 1-99 pct within product-year
gen long hs6_num = real(Product_HS6)
bysort hs6_num Year: egen double p1  = pctile(log_uv), p(1)
bysort hs6_num Year: egen double p99 = pctile(log_uv), p(99)
keep if log_uv >= p1 & log_uv <= p99
drop p1 p99

* Keep only ISR, CHN, IND — CHN and IND are the control group
keep if inlist(Cntry_Org_Code, "ISR", "CHN", "IND")

* Israeli indicator (reference = China + India)
gen byte d_ISR = (Cntry_Org_Code == "ISR")

* Qunt_Unit is already numeric — use directly as unit FE

* Unit value is ImpVal_USD / Quantity — already in USD
* Comparisons will be within product x year x unit cells, so unit-of-measure
* differences cannot confound the Israeli price premium

count if d_ISR == 1
di "Final sample: `=_N' obs,  Israeli: `r(N)'"

* ── regressions ──────────────────────────────────────────────────────────────
* reghdfe handles the FE directly from string/numeric vars — no need to pre-encode

foreach sector in all textiles other {

    preserve

    if "`sector'" == "textiles" keep if hs2 >= 50 & hs2 <= 63
    if "`sector'" == "other"    keep if hs2 <  50 | hs2 >  63

    di _newline "Sector: `sector'  (N=`=_N')"

    * Spec 1: firm x product x year FE, absorbing quantity unit
    * Qunt_Unit absorbed into product x year x unit cell — only same-unit comparisons
    reghdfe log_uv d_ISR, absorb(Trader_ID#hs6_num#Year hs6_num#Year#Qunt_Unit) vce(cluster hs6_num#Year)
    di "Spec1  beta=" %7.4f _b[d_ISR] "  SE=" %6.4f _se[d_ISR] "  ratio=" %6.4f exp(_b[d_ISR])
    estimates store s1_`sector'

    * Spec 2: firm FE + product x year x unit FE
    reghdfe log_uv d_ISR, absorb(Trader_ID hs6_num#Year#Qunt_Unit) vce(cluster hs6_num#Year)
    di "Spec2  beta=" %7.4f _b[d_ISR] "  SE=" %6.4f _se[d_ISR] "  ratio=" %6.4f exp(_b[d_ISR])
    estimates store s2_`sector'

    restore
}

* ── summary tables ───────────────────────────────────────────────────────────
esttab s1_all s1_textiles s1_other, ///
    keep(d_ISR) b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles("All mfg" "Textiles" "Other mfg") ///
    title("Spec 1: firm x product x year FE")

esttab s2_all s2_textiles s2_other, ///
    keep(d_ISR) b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    mtitles("All mfg" "Textiles" "Other mfg") ///
    title("Spec 2: firm FE + product x year FE")

* ── save to CSV ──────────────────────────────────────────────────────────────
file open res using `"`out_dir'\price_wedge_reg_results.csv"', write replace
file write res "spec,sector,beta,se,ratio,N" _n
foreach sector in all textiles other {
    foreach sp in 1 2 {
        estimates restore s`sp'_`sector'
        file write res "`sp',`sector'," ///
            %8.5f (_b[d_ISR]) "," %8.5f (_se[d_ISR]) "," ///
            %8.5f (exp(_b[d_ISR])) "," "`=e(N)'" _n
    }
}
file close res
di "Saved: `out_dir'\price_wedge_reg_results.csv"
