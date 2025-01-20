# pystallment
A Python library for pricing installment options.

Supported products:
* Bermuda options
* Discrete installment options
* European continuous installment options 
* American continuous installment options

Supported methods:
* Discrete analytic formulas
* Binomial model
* Finite Difference (FD)
* Least Squares Monte Carlo (LSMC)
* Inversion of Laplace-Carson transform (LCT)
* Extrapolation of discrete prices (Richardson and polynomial)

| Option Type                     |  Discrete  | Binomial |  FD  |  LSMC  | LCT | Extrapolation |
|:--------------------------------|:----------:|:--------:|:----:|:------:|:---:|:-------------:|
| Bermuda                         |     x      |    x     |      |        |     |               |
| Discrete Installment            |     x      |          |      |        |     |               |
| European Continuous Installment |            |   x      |  x   |   x    |  x  |       x       |
| American Continuous Installment |            |      x   |      |   x    |     |               |


## Installation
`pip install .`

## Usage
First import the modules:
```
from pystallment.algorithms import fdm as fdm
from pystallment import option as opt
```
Second create an appropriate `Option` object:
```
S = 105   # spot
K = 100   # strike
r = 0.05  # riskfree rate
d = 0.04  # dividend yield
T = 1     # time to maturity
q = 3     # installment rate
phi = +1  # +1 for call, -1 for put
opt = ContinuousInstallmentOption(S, K, r, d, vola, T, q, +1)
```
Then create a pricer object and call `price()`:
```
p = FDPricer(opt)
price = p.price()
print(f"price = {price:.5}")
```
## Tests
`pytest tests`
