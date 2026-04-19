# rates-derivatives-pricer

EUR interest rate derivatives pricer built from scratch using NumPy, SciPy, and Pandas.
No pricing library (no QuantLib). Every model is implemented and documented manually.

---

## Motivation

Interest rate derivatives desks require professionals who can both build and validate
pricing models. This project implements the core building blocks of rates structuring:
curve construction, vanilla derivatives pricing, and risk sensitivities.

---

## Project Structure
rates-derivatives-pricer/
├── notebooks/
│   ├── 01_zc_curve_bootstrapping.ipynb
│   ├── 02_cap_floor_pricing.ipynb
│   └── 03_greeks_sensitivity.ipynb
├── src/
│   ├── curve.py
│   ├── capfloor.py
│   └── greeks.py
├── outputs/
├── main.py
├── requirements.txt
└── README.md

---

## Modules

### Module 1 : Zero-Coupon Curve Bootstrapping

Extracts zero-coupon rates and discount factors from EUR par swap rates
using iterative bootstrapping.

**Input:** EUR swap rates on standard maturities (1Y, 2Y, 3Y, 5Y, 7Y, 10Y)

**Method:**

A par swap of maturity $T_N$ with fixed rate $S_N$ satisfies the no-arbitrage condition:

$$S_N \sum_{i=1}^{N} \delta_i \cdot P(0, T_i) + P(0, T_N) = 1$$

Solving iteratively for each discount factor $P(0, T_N)$:

$$P(0, T_N) = \frac{1 - S_N \sum_{i=1}^{N-1} \delta_i \cdot P(0, T_i)}{1 + S_N \cdot \delta_N}$$

Zero-coupon rates are derived as:

$$r_{zc}(T) = -\frac{\ln P(0, T)}{T}$$

The instantaneous forward rate is approximated by central finite differences:

$$f(0, T) \approx -\frac{\ln P(0, T+h) - \ln P(0, T-h)}{2h}$$

Interpolation between pillars uses log-linear interpolation on discount factors,
which guarantees positive forward rates.

**Output:** ZC curve + instantaneous forward curve

---

### Module 2 : Cap and Floor Pricing (Black 1976)

Prices interest rate Caps and Floors as portfolios of Caplets/Floorlets
under the Black (1976) model.

**Method:**

A Cap is decomposed into $N$ Caplets. Each Caplet pays at $T_{i+1}$:

$$\text{Caplet}(T_i, T_{i+1}) = \delta \cdot P(0, T_{i+1}) \cdot
\left[ F_i \cdot N(d_1) - K \cdot N(d_2) \right]$$

where:

$$d_1 = \frac{\ln(F_i / K) + \frac{1}{2} \sigma^2 T_i}{\sigma \sqrt{T_i}}, 
\quad d_2 = d_1 - \sigma \sqrt{T_i}$$

$F_i$ is the simply-compounded forward rate over $[T_i, T_{i+1}]$,
extracted from the Module 1 curve.

Cap-Floor parity (equivalent to put-call parity):

$$\text{Cap} - \text{Floor} = \text{Swap}$$

**Output:** Cap/Floor price + decomposition by caplet

---

### Module 3 : Greeks and Sensitivity Analysis

Computes risk sensitivities of Cap/Floor prices using bump-and-reprice
(finite differences).

| Greek | Definition | Bump |
|-------|-----------|------|
| Delta | Sensitivity to forward rate | +1 bp on $F_i$ |
| Vega  | Sensitivity to implied volatility | +1% on $\sigma$ |
| Rho   | Sensitivity to discount rate | +1 bp on $r_{zc}$ |

**Output:** Greeks table + Cap price surface as a function of strike and maturity

---

## Installation

```bash
git clone https://github.com/Yanael25/rates-derivatives-pricer.git
cd rates-derivatives-pricer
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Or open the notebooks in order:
notebooks/01_zc_curve_bootstrapping.ipynb
notebooks/02_cap_floor_pricing.ipynb
notebooks/03_greeks_sensitivity.ipynb

---

## Dependencies
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
jupyter>=1.0

---

## Author

Yanael ZOHOU
M1 Financial Engineering Student 
