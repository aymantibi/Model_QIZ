# QIZ Heterogeneous-Firm Trade Model (Multi-Sector, Two Regions)

This note lays out a tractable heterogeneous-firm trade model tailored to Qualifying Industrial Zones (QIZ). The model links micro responses—export participation, destination expansion, rules-of-origin (ROO) take-up, and productivity upgrading—to general equilibrium outcomes in Egypt: wages, employment reallocation, domestic price indices, and welfare.

It is designed to be **intuitive but not overly stylized**:
- ROO applies only to **US-destined output**, consistent with shipment-level certification.
- The ROO intermediate bundle is **normalized** to avoid a mechanical cost penalty.
- Compliance is partial via **idiosyncratic compliance costs** (lognormal).
- Productivity upgrading is **triggered by US market access**, and raises performance in all destinations.

---

## 1. Environment

### Regions, sectors, destinations
- Regions: \( r \in \{Q,N\} \), where \(Q\) is QIZ-eligible and \(N\) is non-QIZ.
- Sectors: \( s \in S \). In code we begin with \(S=\{T,O\}\) (Textiles/Apparel, Other manufacturing).
- Destinations: \( j \in \{\text{EG}, \text{US}, \text{RW}\} \) (domestic Egypt, United States, Rest of World).

Egypt is small relative to the US and RW, so foreign shifters \( (E_{js}, P_{js}) \) for \(j \in \{\text{US}, \text{RW}\}\) are taken as exogenous.

---

## 2. Preferences and demand

Consumers in destination \(j\) have nested CES preferences over sectors and varieties.

### Upper tier (across sectors)
\[
U_j = \left( \sum_{s\in S} \beta_s^{1/\eta} \, U_{js}^{\frac{\eta-1}{\eta}} \right)^{\frac{\eta}{\eta-1}},
\]
where \(\beta_s>0\) are sector weights (\(\sum_s \beta_s=1\)) and \(\eta>0\) is the elasticity across sectors.

### Lower tier (within sector)
\[
U_{js} = \left( \int_{\omega\in\Omega_{js}} q_{js}(\omega)^{\frac{\sigma_s-1}{\sigma_s}} \, d\omega \right)^{\frac{\sigma_s}{\sigma_s-1}},
\]
where \(\sigma_s>1\) is the within-sector elasticity.

### CES demand and sector price indices
For a variety \(\omega\) priced at \(p_{js}(\omega)\),
\[
q_{js}(\omega) = \left(\frac{p_{js}(\omega)}{P_{js}}\right)^{-\sigma_s}\frac{E_{js}}{P_{js}},
\]
\[
P_{js} = \left( \int_{\omega\in\Omega_{js}} p_{js}(\omega)^{1-\sigma_s}\, d\omega \right)^{\frac{1}{1-\sigma_s}},
\]
and revenue is:
\[
R_{js}(\omega)=E_{js}\left(\frac{p_{js}(\omega)}{P_{js}}\right)^{1-\sigma_s}.
\]

### Egypt’s endogenous expenditures
Egyptian income:
\[
Y_{\text{EG}} = w_Q L_Q + w_N L_N + T,
\]
where \(T\) is a net transfer that closes the small-open economy.

Sector expenditures follow CES allocation:
\[
E_{\text{EG},s} = \beta_s \left(\frac{P_{\text{EG},s}}{P_{\text{EG}}}\right)^{1-\eta}Y_{\text{EG}},
\]
with aggregate price index:
\[
P_{\text{EG}} = \left(\sum_{s\in S}\beta_s P_{\text{EG},s}^{1-\eta}\right)^{\frac{1}{1-\eta}}.
\]

---

## 3. Firms: entry, technology, costs

### Entry and productivity draws
In each \((r,s)\), a mass \(M_{rs}\) of entrants pay entry cost \(w_r f^E_{rs}\) to draw productivity \(\phi\) from Pareto:
\[
G_s(\phi) = 1 - \left(\frac{\phi_{\min,s}}{\phi}\right)^{\theta_s}, \qquad \phi \ge \phi_{\min,s},
\]
with \(\theta_s > \sigma_s-1\).

### Production
\[
y=\phi\, l^{\alpha_s}m^{1-\alpha_s}.
\]

Cost minimization yields unit cost and marginal cost:
\[
mc_{rs}(\phi;c_{m,s})=\frac{1}{\phi}\cdot
\frac{w_r^{\alpha_s}c_{m,s}^{1-\alpha_s}}{\alpha_s^{\alpha_s}(1-\alpha_s)^{1-\alpha_s}}.
\]

---

## 4. Trade costs, tariffs, pricing

Iceberg trade cost \(d_{rjs}\ge 1\). US exports face MFN tariff \(t^{MFN}_s\) unless the firm complies.

Define the delivered wedge:
\[
\tau_{rjs} = d_{rjs} \times
\begin{cases}
1+t^{MFN}_s, & j=\text{US and noncomplier},\\
1, & j=\text{US and complier},\\
1, & j\in\{\text{EG},\text{RW}\}.
\end{cases}
\]

Delivered marginal cost: \(\widetilde{mc}_{rjs}=\tau_{rjs}mc_{rs}\).

Under monopolistic competition with CES, firms charge a constant markup:
\[
p_{rjs}(\phi)=\mu_s \widetilde{mc}_{rjs}(\phi),\qquad \mu_s=\frac{\sigma_s}{\sigma_s-1}.
\]

---

## 5. Export participation and cutoffs

Serving destination \(j\) requires paying a fixed cost \(w_r f_{rjs}\). Variable profit is:
\[
\pi^{var}_{rjs}(\phi)=\frac{1}{\sigma_s}R_{rjs}(\phi).
\]

Serving \(j\) is profitable iff:
\[
\frac{1}{\sigma_s}R_{rjs}(\phi)\ge w_r f_{rjs}.
\]
This defines productivity cutoffs \(\phi^*_{rjs}\) (exporters are those with \(\phi\ge \phi^*_{rjs}\)).

---

## 6. ROO compliance (QIZ) with normalization fix

### Key modeling choice: ROO applies only to US shipments
Non-US production uses the free-sourcing intermediate unit cost:
\[
c^N_{m,s}=p_{\text{RW},s}.
\]

For US-destined shipments, a compliant QIZ firm must satisfy Israeli content \(\gamma_s\).

### Fix #1 (Normalization)
Define the compliant intermediate composite as a **normalized Cobb–Douglas**:
\[
m \equiv \frac{m_{\text{IL}}^{\gamma_s} m_{\text{RW}}^{1-\gamma_s}}{\gamma_s^{\gamma_s}(1-\gamma_s)^{1-\gamma_s}}.
\]
Then the **derived** unit cost is:
\[
c^{mix}_{m,s}(\gamma_s) = p_{\text{IL},s}^{\gamma_s}p_{\text{RW},s}^{1-\gamma_s}.
\]
This ensures that if \(p_{\text{IL},s}=p_{\text{RW},s}\), compliance does not mechanically raise unit cost.

### Administrative wedge increasing in stringency
To capture verification and coordination costs:
\[
\chi_s(\gamma_s)=1+\xi_s \gamma_s,\qquad \xi_s>0,
\]
so the US-only compliant intermediate cost is:
\[
c^{C,US}_{m,s}(\gamma_s)=\chi_s(\gamma_s)c^{mix}_{m,s}(\gamma_s).
\]

### Partial take-up via idiosyncratic compliance costs
Compliance also requires a fixed cost, heterogeneous across firms:
\[
f^C_{i,s} = f^C_s \exp(\varepsilon_i),\qquad \varepsilon_i \sim N(0,\sigma_{C,s}^2).
\]

A Q firm complies if the profit gain from tariff-free US access exceeds the compliance costs:
\[
\Pi^{C}_{Q,s}(\phi,\varepsilon;\gamma_s) \ge \Pi^{N}_{Q,s}(\phi) .
\]

Because compliance lowers the US wedge but increases US marginal cost, only a subset comply.

---

## 7. Productivity upgrading triggered by US access

Upgrading scales productivity:
\[
\phi'=\delta_s \phi,\qquad \delta_s>1,
\]
at fixed cost \(w_r f^U_s\).

To capture “market access → upgrading” without heavy dynamics, we impose:
> **Upgrading is feasible only if the firm serves the US.**

Then the firm upgrades iff:
\[
\Pi_{rs}(\delta_s\phi;\text{optimal markets and compliance})-\Pi_{rs}(\phi;\text{optimal markets and compliance})\ge w_r f^U_s.
\]

Upgrading raises revenues in all destinations because under CES with markup pricing:
\[
R_{rjs} \propto \phi^{\sigma_s-1}
\quad\Rightarrow\quad
\frac{R'_{rjs}}{R_{rjs}}=\delta_s^{\sigma_s-1}.
\]
Thus upgrading generates spillovers to RW exports for firms that upgrade due to US access.

---

## 8. Labor mobility and regional labor markets

Total labor \(L\). Workers choose region based on real wages (logit):
\[
\lambda_r=
\frac{(w_r/P_{\text{EG}})^\kappa}{\sum_{r'}(w_{r'}/P_{\text{EG}})^\kappa},
\qquad L_r=\lambda_r L,
\]
with \(\kappa>0\).

Labor demand in region \(r\) equals labor used in variable production plus fixed costs and entry across sectors:
\[
L_r = \sum_{s\in S} M_{rs}\mathbb{E}[l^{var}_{rs}(\phi,\varepsilon)+l^{fix}_{rs}(\phi,\varepsilon)] + \sum_{s\in S} M_{rs} f^E_{rs}.
\]

---

## 9. Equilibrium

An equilibrium consists of:
- wages \(\{w_r\}\),
- employment \(\{L_r\}\),
- entry masses \(\{M_{rs}\}\),
- domestic sector price indices \(\{P_{\text{EG},s}\}\) and \(P_{\text{EG}}\),
- firm policies: destination participation, compliance, upgrading,

such that:

1. **Households** maximize utility (implies CES demands).
2. **Firms** choose prices, destinations, compliance, and upgrading to maximize profits.
3. **Free entry:** for each \((r,s)\),
   \[
   \mathbb{E}[\Pi_{rs}(\phi,\varepsilon)] = w_r f^E_{rs}.
   \]
4. **Labor mobility** holds:
   \[
   L_r = \lambda_r L.
   \]
5. **Labor market clearing** in each region.
6. **Price indices** satisfy CES aggregation given active varieties.

Welfare:
\[
W=\frac{Y_{\text{EG}}}{P_{\text{EG}}}.
\]

---

## 10. Counterfactuals

### (A) Shut down productivity channel
Set \(\delta_s=1\) (or make \(f^U_s\) prohibitively large). Re-solve equilibrium and compare:
- compliance,
- exports to US and RW,
- wages and welfare.

Interpretation: without upgrading, the model cannot generate the empirical pattern that compliers increase RW exports and experience productivity gains.

### (B) Vary ROO content requirement \(\gamma_s\)
Solve equilibrium for a grid of \(\gamma_T\in[0,\bar{\gamma}_T]\) holding everything else fixed. For each \(\gamma_T\), compute:
- compliance rate,
- US and RW exports,
- Israeli input intensity (implied by \(\gamma_T\)),
- wages, employment shares, welfare \(W(\gamma_T)\).

Plotting \(W(\gamma_T)\) traces the policy frontier (conditionality vs welfare gains).

---

## 11. Practical calibration notes (defensible starting values)

- \(\sigma_T\approx 6\text{–}7\) for textiles; \(\sigma_O\approx 4\).
- \(t_T^{MFN}\approx 0.10\text{–}0.20\).
- \(\gamma_T\approx 0.105\).
- Israeli price premium \(p_{IL}/p_{RW}\approx 1.05\text{–}1.15\).
- Admin wedge slope \(\xi_T\approx 0.5\text{–}1.5\).
- Compliance cost dispersion \(\sigma_C\approx 0.3\text{–}0.7\).
- Upgrading \(\delta_T\approx 1.10\text{–}1.20\), upgrade cost \(f^U_T\approx 1\text{–}3\).
- Mobility \(\kappa\approx 1\text{–}4\).

The accompanying Python code implements the equilibrium computation and the counterfactuals.
