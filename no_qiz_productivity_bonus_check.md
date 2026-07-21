# No QIZ Productivity Bonus Check

Diagnostic question: can the reformed model fit the data if QIZ compliance does
not add an extra upgrade/productivity complementarity?

Specification for this check:

- Upgrading is allowed with and without QIZ.
- `upgrade_psi_comp_T = 0.0`.
- `upgrade_cost_comp_mult_T = 1.0`.
- Compliance heterogeneity and the MFN-incumbency margin are kept.

Target moments:

- Textile QIZ compliance among US exporters: `0.321`.
- Non-US export growth among textile Israel importers/compliers: `109.8%`.

Selected targeted-grid results:

| mfn incumbency | baseline upgrade return | upgrade cost quad | fC_T | T comp / US exporters | no-QIZ upgrade | QIZ upgrade | complier non-US change | textile non-US change | welfare gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 0.06 | 2.0 | 8.0 | 0.314 | 1.149 | 1.149 | 0.0% | 1.3% | 0.281% |
| 1.5 | 0.06 | 4.0 | 10.0 | 0.343 | 0.666 | 0.666 | 0.0% | 1.2% | 0.283% |
| 2.0 | 0.06 | 4.0 | 8.0 | 0.343 | 0.761 | 0.761 | 0.0% | 1.3% | 0.288% |
| 2.0 | 0.03 | 4.0 | 6.0 | 0.361 | 0.316 | 0.316 | 0.0% | 0.0% | 0.294% |

Conclusion:

The model can approximately match the textile compliance rate without the QIZ
productivity complementarity. However, it does not match the non-US export
growth mechanism. Around the compliance target, aggregate upgrading is unchanged
from no-QIZ to QIZ, and same-type complier non-US export growth is essentially
zero. The QIZ route changes US market access, but without an added productivity
or learning channel it does not generate the observed non-US export increase.
