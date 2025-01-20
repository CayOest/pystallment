# pystallment
A Python library for pricing installment options.

Supported products:
* Bermuda options
* Discrete installment options
* European continuous installment options 
* American continuous installment options

Supported methods:
* Discrete analytic formulas
* Extrapolation of discrete prices (Richardson and polynomial)
* Binomial model
* Finite Difference
* Least Squares Monte Carlo
* Inversion of Laplace-Carson transform

|          |  Discrete  | Binomial |  FD  |  LSMC  | LCT | Extrapolation |
|:---------|:----------:|:--------:|:----:|:------:|:---:|:-------------:|
| Bermuda  |     x      |    x     |      |        |     |               |
| n-Inst   |     x      |          |      |        |     |               |
| EU Inst  |            |   x      |  x   |   x    |  x  |       x       |
|  Am Inst |            |      x   |      |   x    |     |               |


## Installation
`pip install .`

## Tests
`pytest tests`