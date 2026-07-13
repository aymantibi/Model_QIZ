# Understanding the QIZ Heterogeneous-Firm Trade Model
## A Personal Study Guide

---

## 1. Overview

### What is this model trying to do?

The model asks: **what happens to Egyptian firms and workers when Egypt gets preferential access to the US market through Qualifying Industrial Zones (QIZs)?**

QIZs are special industrial zones in Egypt (and Jordan) where firms can export to the US **tariff-free**, but only if their products contain a minimum share of Israeli inputs — this is the **Rules of Origin (ROO)** requirement. The model tries to capture:

- Which firms choose to comply with the ROO (and thus export tariff-free to the US)
- Which firms upgrade their productivity because they gain US market access
- How wages, employment, and welfare in Egypt change as a result

### The broad structure

The model has three nested layers:

```
OUTER LOOP (General Equilibrium)
│   Solves for: wages {w_Q, w_N} and firm entry masses {M_rs}
│
└── GOODS BLOCK (Price indices and expenditure)
        Solves for: domestic sector price indices {P_EG,s}
        Given wages and entry masses, computes:
        - Labor allocation across regions
        - Egyptian income and sector expenditures
        - Price indices from aggregating over all active firms
        │
        └── FIRM PROBLEM (innermost)
                For every (phi, eps) draw: choose
                - Which markets to serve (EG, US, RW)
                - Whether to comply with ROO (Q firms only)
                - Whether to upgrade productivity
```

### Regions, sectors, destinations

| Symbol | Meaning |
|--------|---------|
| `Q` | QIZ-eligible region |
| `N` | Non-QIZ region |
| `T` | Textiles/Apparel sector |
| `O` | Other manufacturing sector |
| `EG` | Domestic Egypt market |
| `US` | United States market |
| `RW` | Rest of World market |

Egypt is modeled as a **small open economy**: it takes foreign prices and expenditures as given. What happens in Egypt doesn't move US or RW price indices.

---

## 2. Demand & Price Indices

### The consumer's problem

Egyptian consumers (and foreign consumers in US and RW) have **nested CES preferences** — a two-level structure.

**Upper tier** — how consumers split spending across sectors:

$$U_j = \left(\sum_{s \in S} \beta_s^{1/\eta} \, U_{js}^{\frac{\eta-1}{\eta}}\right)^{\frac{\eta}{\eta-1}}$$

Think of this like a basket of "textiles" and "other goods." The parameter $\eta$ controls how easily consumers substitute between sectors. $\beta_s$ is the budget share weight for sector $s$ (they sum to 1).

**Lower tier** — within each sector, consumers love variety (Dixit-Stiglitz):

$$U_{js} = \left(\int_{\omega \in \Omega_{js}} q_{js}(\omega)^{\frac{\sigma_s - 1}{\sigma_s}} d\omega\right)^{\frac{\sigma_s}{\sigma_s-1}}$$

Here $\sigma_s > 1$ is the **elasticity of substitution** between varieties within sector $s$. Higher $\sigma_s$ means varieties are closer substitutes — consumers are more price-sensitive.

### Deriving the demand for a single variety

To find how much of variety $\omega$ a consumer buys, maximize the lower-tier utility subject to a sector budget $E_{js}$. The result is:

$$q_{js}(\omega) = \left(\frac{p_{js}(\omega)}{P_{js}}\right)^{-\sigma_s} \frac{E_{js}}{P_{js}}$$

**Intuition:** demand for variety $\omega$ falls when its own price $p_{js}(\omega)$ rises (elasticity $-\sigma_s$), and falls when the sector price index $P_{js}$ falls (making other varieties relatively cheaper).

### The CES price index — where does it come from?

The price index $P_{js}$ is defined as the **minimum cost of one unit of sector $s$ utility** in destination $j$. Plugging the optimal demand back into the budget constraint gives:

$$P_{js} = \left(\int_{\omega \in \Omega_{js}} p_{js}(\omega)^{1-\sigma_s} d\omega\right)^{\frac{1}{1-\sigma_s}}$$

**Key property:** $P_{js}$ falls when more varieties enter (more competition lowers the cost-of-living) or when existing varieties lower their prices.

In the code, the price index is built up numerically. Since $1 - \sigma_s < 0$ (because $\sigma_s > 1$), each variety's contribution is $p^{1-\sigma_s}$, which is **smaller** when the price is **higher** — expensive varieties contribute less to the index.

```python
# From solve_goods_block: accumulating P^{1-sigma} contributions
if best["serve"]["EG"]:
    mc = (1.0 / eff_phi) * ((w[r] ** alpha_loc) * (cN_loc ** (1.0 - alpha_loc))) / denom_mc
    pEG = mu_loc * p["d_iceberg"][(r, "EG", s)] * mc
    contrib += wt * (pEG ** (1.0 - sigma))   # <-- this is p^{1-sigma}

P_new[s] = ces_price_from_power(P_pow, sigma)  # raises to 1/(1-sigma)
```

### Revenue of a single variety

Revenue = price × quantity:

$$R_{js}(\omega) = p_{js}(\omega) \cdot q_{js}(\omega) = E_{js} \left(\frac{p_{js}(\omega)}{P_{js}}\right)^{1-\sigma_s}$$

Note: $1 - \sigma_s < 0$, so **higher price → lower revenue share**. This is the standard CES demand system.

### Aggregate price index across sectors

$$P_{\text{EG}} = \left(\sum_{s \in S} \beta_s P_{\text{EG},s}^{1-\eta}\right)^{\frac{1}{1-\eta}}$$

In the code:

```python
def ces_aggregate_price(P_by_s, beta, eta):
    power = 1.0 - eta
    inside = sum(beta[s] * (P_by_s[s] ** power) for s in beta)
    return float(inside ** (1.0 / power))
```

The special case $\eta = 1$ (Cobb-Douglas across sectors) is handled separately using the log formula: $P = \prod_s P_s^{\beta_s}$.

### Sector expenditures

Egyptian income $Y_{\text{EG}} = w_Q L_Q + w_N L_N + T$. Upper-tier CES demand gives:

$$E_{\text{EG},s} = \beta_s \left(\frac{P_{\text{EG},s}}{P_{\text{EG}}}\right)^{1-\eta} Y_{\text{EG}}$$

**Intuition:** if sector $s$ gets relatively cheaper (its price index falls), the expenditure share on that sector rises (when $\eta > 1$, sectors are substitutes).

---

## 3. Firm Technology & Unit Costs

### Production function

Each firm produces using labor $l$ and an intermediate input bundle $m$:

$$y = \phi \cdot l^{\alpha_s} \cdot m^{1-\alpha_s}$$

Here:
- $\phi$ is the firm's **productivity** (higher = more output per input)
- $\alpha_s$ is the labor share in production
- $1 - \alpha_s$ is the intermediate input share

### Deriving marginal cost

The firm minimizes cost $w_r \cdot l + c_m \cdot m$ subject to producing $y$ units.

**Step 1 — input demand.** The first-order conditions give the cost-minimizing input ratio:

$$\frac{l}{m} = \frac{\alpha_s}{1-\alpha_s} \cdot \frac{c_m}{w_r}$$

**Step 2 — total cost.** Substituting back:

$$C(y) = \frac{1}{\phi} \cdot \frac{w_r^{\alpha_s} \cdot c_m^{1-\alpha_s}}{\alpha_s^{\alpha_s}(1-\alpha_s)^{1-\alpha_s}} \cdot y$$

The denominator $\alpha_s^{\alpha_s}(1-\alpha_s)^{1-\alpha_s}$ is a constant that comes from the algebra of Cobb-Douglas cost minimization.

**Step 3 — marginal cost.** Since cost is linear in $y$, marginal cost equals average cost:

$$mc(\phi; c_m) = \frac{1}{\phi} \cdot \frac{w_r^{\alpha_s} \cdot c_m^{1-\alpha_s}}{\alpha_s^{\alpha_s}(1-\alpha_s)^{1-\alpha_s}}$$

In the code:

```python
alpha = p["alpha"][s]
denom = (alpha ** alpha) * ((1.0 - alpha) ** (1.0 - alpha))
mc_base = (1.0 / (phi * delta)) * ((w_r ** alpha) * (cN ** (1.0 - alpha))) / denom
```

Here `cN = p_rw[s]` is the price of the freely-sourced intermediate input, and `delta` is the upgrading multiplier (1 if no upgrade). Note that $mc \propto 1/\phi$: **more productive firms have lower marginal cost**.

### Markup pricing

Under monopolistic competition with CES demand, every firm charges a **constant markup** over marginal cost:

$$p = \mu_s \cdot mc, \qquad \mu_s = \frac{\sigma_s}{\sigma_s - 1}$$

**Why?** The firm faces demand $q = A \cdot p^{-\sigma_s}$ (where $A$ collects things outside the firm's control). Revenue is $R = p \cdot q = A \cdot p^{1-\sigma_s}$. Profit is $\pi = R - mc \cdot q = A(p^{1-\sigma_s} - mc \cdot p^{-\sigma_s})$. Setting $d\pi/dp = 0$:

$$A(1-\sigma_s)p^{-\sigma_s} + A \sigma_s \cdot mc \cdot p^{-\sigma_s - 1} = 0$$

Solving: $p = \frac{\sigma_s}{\sigma_s - 1} mc = \mu_s \cdot mc$.

The markup $\mu_s$ is **higher when $\sigma_s$ is lower** (less substitutable varieties → more market power). For textiles with $\sigma_T = 6.7$: $\mu_T = 6.7/5.7 \approx 1.175$ (17.5% markup). For other manufacturing with $\sigma_O = 4.0$: $\mu_O = 4/3 \approx 1.33$ (33% markup).

### Variable profit

A useful fact from CES: for any destination $j$,

$$\pi^{var}_{rjs} = R_{rjs} - mc \cdot q_{rjs} = R_{rjs} - \frac{R_{rjs}}{\mu_s} = \frac{R_{rjs}}{\sigma_s}$$

**Variable profit is always $1/\sigma_s$ of revenue.** This is why in the code you see `pi_EG = R_EG / sigma` etc.

---

## 4. Trade Costs, Tariffs & Pricing

### Iceberg trade costs

Shipping goods incurs an **iceberg cost**: to deliver 1 unit, the firm must ship $d_{rjs} \geq 1$ units. So the effective marginal cost of delivering to destination $j$ is $d_{rjs} \cdot mc$.

This is called "iceberg" because the melted portion $d_{rjs} - 1$ represents goods lost in transit (or equivalently, the resources used up in shipping).

### Tariffs for US exports

Non-compliant firms exporting to the US face an **MFN (Most Favored Nation) tariff** $t^{MFN}_s$ on top of the iceberg cost. The total delivered wedge is:

$$\tau_{rjs} = d_{rjs} \times \begin{cases} 1 + t^{MFN}_s & j = \text{US, non-compliant} \\ 1 & j = \text{US, compliant (QIZ)} \\ 1 & j \in \{\text{EG, RW}\} \end{cases}$$

Compliant QIZ firms pay zero tariff — that's the whole point of the QIZ arrangement.

### Delivered price

The firm prices at markup over **delivered** marginal cost:

$$p_{rjs}(\phi) = \mu_s \cdot \tau_{rjs} \cdot mc_{rs}(\phi)$$

In the code:

```python
p_EG = mu * (tau_EG * mc_base)
p_RW = mu * (tau_RW * mc_base)
p_US = mu * (tau_US * mc_US)   # mc_US differs for compliant firms
```

Note that `mc_US` can differ from `mc_base` for compliant firms, because compliance changes the intermediate input used for US production (see Section 6).

---

## 5. Export Participation

### The cutoff logic

Serving each destination requires paying a **fixed cost** $w_r f_{rjs}$ (in labor units). A firm serves destination $j$ if and only if variable profit covers the fixed cost:

$$\frac{R_{rjs}(\phi)}{\sigma_s} \geq w_r f_{rjs}$$

Since $R_{rjs} \propto \phi^{\sigma_s - 1}$ (more productive firms have lower prices and thus higher revenue shares), there exists a **productivity cutoff** $\phi^*_{rjs}$ such that only firms with $\phi \geq \phi^*_{rjs}$ find it profitable to serve market $j$.

In the code, this is checked directly:

```python
serve_EG = (pi_EG >= w_r * f_dom)
serve_US = (pi_US >= w_r * f_US)
serve_RW = (pi_RW >= w_r * f_RW)
```

### Market hierarchy

In practice, the domestic market (EG) is the easiest to serve (lowest fixed cost, no trade costs). Exporting to US or RW requires additional fixed costs and iceberg losses. This naturally generates a hierarchy: **all exporters also serve the domestic market**, but not all domestic firms export.

### Productivity draws — the Pareto distribution

Each entrant draws productivity $\phi$ from a Pareto distribution:

$$G_s(\phi) = 1 - \left(\frac{\phi_{\min,s}}{\phi}\right)^{\theta_s}, \qquad \phi \geq \phi_{\min,s}$$

The shape parameter $\theta_s$ must satisfy $\theta_s > \sigma_s - 1$ to ensure that the distribution of firm size has finite variance. Higher $\theta_s$ means more firms cluster near the minimum — a fatter left tail, more low-productivity firms.

In the code, the Pareto grid is approximated numerically with a midpoint rule:

```python
def pareto_grid(phi_min, theta, n):
    u = (np.arange(n) + 0.5) / n       # uniform draws in (0,1)
    phi = phi_min * (1.0 - u) ** (-1.0 / theta)   # inverse CDF of Pareto
    w = np.full(n, 1.0 / n)            # equal probability weights
    return phi, w
```

The `u` values are evenly spaced interior points, then transformed through the inverse Pareto CDF. The weights are uniform because the $u$ grid is uniform over probability mass.

---

## 6. ROO Compliance (QIZ)

This is one of the most novel and subtle parts of the model.

### What is the ROO?

The QIZ agreement says: to export to the US tariff-free, your product must contain at least $\gamma_s$ fraction of Israeli inputs. This is the **Rules of Origin** requirement. It creates a trade-off:

- **Benefit**: pay zero US tariff instead of $t^{MFN}_s$
- **Cost**: use more expensive Israeli inputs (instead of free-market sourcing), plus administrative burden

### Who can comply?

Only firms in the **QIZ-eligible region** (`r = Q`). Non-QIZ firms (`r = N`) cannot access the preferential rate regardless.

### The intermediate input cost under compliance

A compliant firm must use an input bundle with at least fraction $\gamma_s$ Israeli content. Model this as a **Cobb-Douglas mix**:

$$m = \frac{m_{\text{IL}}^{\gamma_s} \cdot m_{\text{RW}}^{1-\gamma_s}}{\gamma_s^{\gamma_s}(1-\gamma_s)^{1-\gamma_s}}$$

The denominator is a **normalization constant** $K = \gamma_s^{\gamma_s}(1-\gamma_s)^{1-\gamma_s}$. This is crucial.

**Why normalize?** Without normalization, increasing $\gamma_s$ from 0 to any positive value would mechanically inflate the unit cost just because of the functional form change. The normalization ensures that when $p_{\text{IL}} = p_{\text{RW}}$ (Israeli and world inputs cost the same), compliance has **zero cost in terms of unit cost**. The ROO only bites when Israeli inputs are genuinely more expensive.

**Deriving the unit cost of the normalized mix.** The unit cost of producing one unit of the normalized composite is found by cost minimization:

$$c^{mix}_{m,s}(\gamma_s) = p_{\text{IL},s}^{\gamma_s} \cdot p_{\text{RW},s}^{1-\gamma_s}$$

This is a weighted geometric mean of input prices. The normalization constant $K$ in the quantity index cancels out when you derive the cost, leaving this clean formula.

**Verification:** if $p_{\text{IL}} = p_{\text{RW}} = p$, then $c^{mix} = p^{\gamma_s} \cdot p^{1-\gamma_s} = p$. Cost equals the common input price — no compliance penalty. ✓

In the code:

```python
def cmix_normalized(p_il, p_rw, gamma):
    return (p_il ** gamma) * (p_rw ** (1.0 - gamma))
```

### The administrative wedge

Beyond input costs, compliance creates paperwork and coordination costs. These scale with the stringency $\gamma_s$:

$$\chi_s(\gamma_s) = 1 + \xi_s \cdot \gamma_s$$

So the **total compliant intermediate cost for US shipments** is:

$$c^{C,US}_{m,s} = \chi_s(\gamma_s) \cdot c^{mix}_{m,s}(\gamma_s)$$

And the compliant marginal cost for US production:

$$mc^C_{US} = \frac{1}{\phi} \cdot \frac{w_r^{\alpha_s} \cdot (c^{C,US}_{m,s})^{1-\alpha_s}}{\alpha_s^{\alpha_s}(1-\alpha_s)^{1-\alpha_s}}$$

**Key point**: compliance raises the US marginal cost but removes the tariff wedge $1 + t^{MFN}_s$. A firm complies only if the tariff saving outweighs the cost increase.

### ROO applies only to US shipments

This is a deliberate and important modeling choice. Non-US production (Egypt domestic, RW exports) uses the free-market input $c^N_m = p_{\text{RW},s}$. Only US-destined output must satisfy the Israeli content requirement. This is consistent with how QIZ certification works in practice: it's shipment-level.

### Partial compliance — idiosyncratic fixed costs

Even among firms that would benefit from compliance in expectation, not all comply. Each firm draws an idiosyncratic compliance cost:

$$f^C_{i,s} = f^C_s \cdot \exp(\varepsilon_i), \qquad \varepsilon_i \sim N(0, \sigma_{C,s}^2)$$

So $f^C_{i,s}$ is **lognormally distributed**: the median compliance cost is $f^C_s$ but some firms face much higher costs (e.g., their supply chain is hard to restructure). This generates **partial take-up**: some firms comply, some don't, even among those serving the US.

In the code, `eps` represents a draw from $N(0,1)$ and the compliance cost is:

```python
fC_i = p["fC_mean"][s] * np.exp(p["sigma_C"][s] * eps)
```

> **Calibration note:** `fC_mean` is the **median** of this lognormal distribution, not the mean. The true mean is `fC_mean * exp(sigma_C^2 / 2)`. For textiles: mean ≈ `0.35 * exp(0.125) ≈ 0.397`. This matters if you're trying to match average compliance expenditure from data.

### The compliance decision in the code

The code enumerates four strategies for Q-region firms: `(comply=False, upgrade=False)`, `(comply=False, upgrade=True)`, `(comply=True, upgrade=False)`, `(comply=True, upgrade=True)`. For each, it computes the total profit and picks the best. This is a discrete choice over strategy combinations.

---

## 7. Productivity Upgrading

### The mechanism

Serving the US market exposes firms to demanding buyers, quality standards, and global value chains — this triggers **productivity upgrading**. The model captures this by letting any firm that serves the US pay a fixed cost $w_r f^U_s$ to scale its productivity:

$$\phi' = \delta_s \cdot \phi, \qquad \delta_s > 1$$

### Why does upgrading help across all markets?

Under CES demand with markup pricing, revenue in any destination $j$ satisfies:

$$R_{rjs} \propto \phi^{\sigma_s - 1}$$

(because lower $mc \propto 1/\phi$ translates to lower price, which expands demand). After upgrading, $\phi' = \delta_s \phi$, so:

$$\frac{R'_{rjs}}{R_{rjs}} = \delta_s^{\sigma_s - 1}$$

This ratio is the **same for all destinations**. Upgrading lifts revenues everywhere — including in Egypt and RW — because lower prices expand market share under CES. This is the **spillover mechanism**: firms that upgrade because of US access also become more competitive in third markets (RW).

### The upgrade constraint

Upgrading is only available if the firm serves the US (`upgrade_requires_US = True`). In the code:

```python
if upgrade and p["upgrade_requires_US"] and (not serve_US):
    eff_upgrade = False   # can't upgrade without US access
```

If a firm would upgrade but doesn't end up serving the US (because even with the upgrade, the US isn't profitable enough), the upgrade is dropped and profits are recomputed without it.

---

## 8. General Equilibrium

### Labor mobility

Workers choose between regions `Q` and `N` based on real wages. The **logit** allocation is:

$$\lambda_r = \frac{(w_r / P_{\text{EG}})^\kappa}{\sum_{r'} (w_{r'} / P_{\text{EG}})^\kappa}, \qquad L_r = \lambda_r \cdot L$$

Higher $\kappa$ means workers are more responsive to wage differences (more mobile). At $\kappa \to \infty$, wages equalize. At $\kappa = 0$, labor is immobile.

### Free entry

In each $(r, s)$, firms pay entry cost $w_r f^E_{rs}$ to draw a productivity. In equilibrium, the **expected profit from entry equals the entry cost**:

$$\mathbb{E}[\Pi_{rs}(\phi, \varepsilon)] = w_r f^E_{rs}$$

This pins down the mass of entrants $M_{rs}$. If expected profits exceed entry cost, more firms enter, driving down profits (via the price index) until equality holds.

### Labor market clearing

In each region, total labor demand must equal labor supply:

$$L_r = \sum_{s \in S} M_{rs} \left[\mathbb{E}[l^{var}_{rs}] + \mathbb{E}[l^{fix}_{rs}] + f^E_{rs}\right]$$

where:
- $\mathbb{E}[l^{var}_{rs}]$ = expected variable labor (from production)
- $\mathbb{E}[l^{fix}_{rs}]$ = expected fixed labor (domestic, export, compliance, upgrade costs)
- $f^E_{rs}$ = entry cost per entrant

### The outer loop solver

The solver updates wages and entry masses using a simple gradient step:

```python
# If Ld > Ls: labor is scarce -> raise wage
w[r] *= exp(step * log(Ld[r] / Ls[r]))

# If E[profit] > w*fE: entry is too profitable -> more firms enter
M[(r,s)] *= exp(step * log(E_profit / entry_cost))
```

This is a **log-linear tatonnement**: the system adjusts wages up when labor demand exceeds supply, and entry masses up when profits exceed entry costs. The step size `outer_step = 0.10` controls how fast it adjusts (smaller = more stable, slower).

### Welfare

Welfare is real income:

$$W = \frac{Y_{\text{EG}}}{P_{\text{EG}}}$$

This is the standard measure: how much aggregate utility (in units of the composite good) Egyptian households can afford.

---

## 9. Counterfactuals

### Counterfactual A — Shut down the productivity channel

Set $\delta_s = 1$ for all sectors (no upgrading benefit). Re-solve the full GE. Compare:

- **Compliance rates**: without upgrading, the only benefit of compliance is tariff-free US access. Compliance rates may fall.
- **RW exports**: without the upgrading spillover, compliant firms won't gain RW competitiveness. If baseline shows RW exports rising with compliance, this counterfactual tests whether upgrading drives it.
- **Wages and welfare**: if upgrading is important, shutting it down should reduce wages in Q and aggregate welfare.

In the code:

```python
p_off["delta"] = {s: 1.0 for s in p["sectors"]}
```

### Counterfactual B — Vary the ROO content requirement $\gamma_T$

Solve GE for a grid of $\gamma_T \in [0, \bar{\gamma}_T]$, holding everything else fixed. For each $\gamma_T$, record welfare, wages, compliance rates, US exports, RW exports.

**What to expect:**
- At $\gamma_T = 0$: no Israeli content required. Compliance is cheap (no input cost change, just paperwork). High compliance rate.
- As $\gamma_T$ increases: compliance becomes more costly (more Israeli inputs needed, more admin burden). Fewer firms comply. But those that do use more Israeli content.
- Welfare $W(\gamma_T)$ traces a **policy frontier**: the trade-off between Israeli conditionality and Egyptian welfare gains.

---

## 10. Bugs & Fixes

The following issues were identified in `qiz_model_ge.py`. Each is briefly described with its location and correction.

---

### Bug 1 — `serve_EG` not rechecked after upgrade fallback *(High severity)*

**Location:** `firm_best`, lines ~286–321

**What happens:** When a firm plans to upgrade (`upgrade=True`) but then fails to serve the US (so the upgrade is dropped), the code recomputes `R_EG` under the non-upgrade marginal cost. However, `serve_EG` is never updated to reflect the higher (non-upgraded) cost. A firm that was only marginally active in Egypt because of the upgrade boost can be incorrectly recorded as active after the upgrade is dropped.

**Economic consequence:** Corrupts the domestic price index (a non-viable variety enters it), overstates labor demand in Egypt, and inflates free-entry expected profits.

**Fix:** After the upgrade fallback block, add:
```python
pi_EG_new = R_EG / sigma
serve_EG = (pi_EG_new >= w_r * f_dom)
if not serve_EG:
    profit = 0.0
    serve_US = False
    serve_RW = False
    eff_upgrade = False
    eff_comply  = False
```

---

### Bug 2 — `normal_grid` uses endpoint-inclusive `linspace`, not a true midpoint rule *(Medium severity)*

**Location:** `normal_grid`, lines ~154–162

**What happens:** The docstring says "midpoint-rule quadrature" but `np.linspace(-a, a, n)` includes the endpoints $-a$ and $+a$. A true midpoint rule uses only interior points. The step size `h = 2a/(n-1)` is also the linspace spacing, not the midpoint spacing `2a/n`. The subsequent normalization `w = w/w.sum()` partially corrects for this but the nodes are slightly off.

**Fix:** Replace:
```python
x = np.linspace(-a, a, n)
h = 2.0 * a / (n - 1) if n > 1 else 1.0
```
with:
```python
x = -a + (np.arange(n) + 0.5) * (2.0 * a / n)
h = 2.0 * a / n
```

---

### Bug 3 — `print_key` references `p` which is not in scope *(High severity)*

**Location:** `print_key`, line ~637

**What happens:** `print_key(sol, label)` uses `p["L_total"]` but `p` is not a parameter of the function. It works only because `p` is defined globally in `__main__`. If called from any other context, it raises `NameError`.

**Fix:** Add `p` as a parameter:
```python
def print_key(sol: Dict[str, Any], label: str, p: Dict[str, Any]):
    ...
    print("Employment shares:", {r: sol["Ls"][r]/p["L_total"] for r in sol["Ls"]})
```

---

### Bug 4 — Line 636 in `print_key` always prints 1.0 *(Medium severity)*

**Location:** `print_key`, line ~636

**What happens:**
```python
{r: sol["Ls"][r]/sol["goods"]["Ls"][r] * (...) for r in sol["Ls"]}
```
`sol["Ls"]` and `sol["goods"]["Ls"]` are the **same Python object** (assigned as `Ls = goods["Ls"]` in the solver). So this always evaluates to `1.0`. The comment even says "placeholder." This line should be removed.

---

### Bug 5 — Compliance-without-US strategy uses sentinel `-1e18` *(Medium severity — fragile design)*

**Location:** `firm_best`, lines ~328–337

**What happens:** When `comply=True` but `serve_US=False`, the code sets `profit = -1e18` to ensure this strategy never wins. This works because `(comply=False, upgrade=False)` is always in the strategy set. But it's fragile: any future refactoring that reorders or removes strategies could silently break this.

**Suggested fix:** Skip the strategy entirely with `continue` rather than using a sentinel:
```python
if comply and (r == "Q") and qiz_on and (not serve_US):
    continue   # compliance without US access is dominated; skip
```

---

### Bug 6 — `counterfactual_shutdown_productivity` uses a shallow copy *(Low severity — fragile)*

**Location:** `counterfactual_shutdown_productivity`, line ~600

**What happens:** `p_off = {k: v for k, v in p.items()}` is a shallow copy. It is currently safe because `p_off["delta"]` is immediately reassigned to a new dict. But if anyone adds `p_off["delta"]["T"] = 1.0` (in-place mutation) instead, it silently corrupts the original `p`.

**Fix:** Use `import copy; p_off = copy.deepcopy(p)`.

---

### Note — `fC_mean` is the median, not mean, of the compliance cost distribution

**Location:** `params_defensible`, lines ~109–110

**What it means:** `fC_i = fC_mean * exp(sigma_C * eps)` defines a lognormal where `fC_mean` is the **median**. The **mean** is `fC_mean * exp(sigma_C^2 / 2)`. For textiles: `0.35 * exp(0.125) ≈ 0.397`. If calibrating to average compliance expenditure data, the target should be the mean, not the median.

---

### Note — No explicit numeraire for wages

**Location:** `solve_equilibrium`, line ~521

**What it means:** Wages are initialized at `w_Q = w_N = 1.0` but no wage is held fixed during iteration. The model is implicitly pinned by `P_foreign = 1.0`. If foreign prices are ever changed, the absolute wage level becomes unanchored. Best practice: after each update step, normalize one wage (e.g., `w["Q"] = 1.0`) and rescale accordingly.
