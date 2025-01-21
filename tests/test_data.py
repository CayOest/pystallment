kimura_inst_call = [
    # (q, S, Euler, Gaver) with T = 1, K = 100, r = 0.03, d = 0.05, vola = 0.2
(1,	95,	3.7071,	3.6841),
(1, 105,	8.3994,	8.3871),
(1, 115,	14.853,	14.8471),
(3,	95,	2.228,	2.2039),
(3, 105,	6.6385,	6.621),
(3, 115,	12.9687,	12.9585),
(6,	95,	0.6754,	0.684),
(6, 105,	4.2745,	4.2725),
(6, 115,	10.2533,	10.2489)
]


anton_inst_call = [
# (q, S, Gaver, Krishni) with T = 1, K = 100, r = 0.03, d = 0.05, vola = 0.2
(1,	95,	3.7072,	3.7069),
(1,	105,	8.3995,	8.3993),
(1,	115,	14.8531,	14.8531),
(3,	95,	2.2283,	2.2266),
(3,	105,	6.6388,	6.6379),
(3,	115,	12.969,	12.9686),
(6,	95,	0.6761,	0.6703),
(6,	105,	4.2752,	4.2723),
(6,	115,	10.254,	10.2527)

]

ciurlia_inst_call = [
    # (vola, S, T, q, CNFD) with K = 100, r = 0.05, d = 0.04
    (0.2, 96, 0.25, 1, 2.0671),
    (0.2, 96, 0.25, 3, 1.6607),
    (0.2, 96, 0.25, 8, 0.8035),
    (0.2, 96, 1, 1, 5.2323),
    (0.2, 96, 1, 3, 3.6495),
    (0.2, 96, 1, 8, 0.7154),
    (0.2, 100, 0.25, 1, 3.8343),
    (0.2, 100, 0.25, 3, 3.3845),
    (0.2, 100, 0.25, 8, 2.3378),
    (0.2, 100, 1, 1, 7.1999),
    (0.2, 100, 1, 3, 5.5107),
    (0.2, 100, 1, 8, 1.9836),
    (0.2, 104, 0.25, 1, 6.2295),
    (0.2, 104, 0.25, 3, 5.7558),
    (0.2, 104, 0.25, 8, 4.6076),
    (0.2, 104, 1, 1, 9.4775),
    (0.2, 104, 1, 3, 7.714),
    (0.2, 104, 1, 8, 3.7678),
    # vola = 0.3
    (0.3, 96, 0.25, 1, 3.8976),
    (0.3, 96, 0.25, 3, 3.4668),
    (0.3, 96, 0.25, 8, 2.4794),
    (0.3, 96, 1, 1, 8.8982),
    (0.3, 96, 1, 3, 7.2435),
    (0.3, 96, 1, 8, 3.6802),
    (0.3, 100, 0.25, 1, 5.8017),
    (0.3, 100, 0.25, 3, 5.3476),
    (0.3, 100, 0.25, 8, 4.2685),
    (0.3, 100, 1, 1, 10.9795),
    (0.3, 100, 1, 3, 9.2661),
    (0.3, 100, 1, 8, 5.4322),
    (0.3, 104, 0.25, 1, 8.1256),
    (0.3, 104, 0.25, 3, 7.6559),
    (0.3, 104, 0.25, 8, 6.5161),
    (0.3, 104, 1, 1, 13.2649),
    (0.3, 104, 1, 3, 11.5044),
    (0.3, 104, 1, 8, 7.4560),
]

ciurlia_inst_call_short = [
    # (vola, S, T, q, CNFD) with K = 100, r = 0.05, d = 0.04
    (0.2, 96, 1, 1, 5.2323),
    (0.2, 96, 1, 3, 3.6495),
    (0.2, 96, 1, 8, 0.7154),
    (0.2, 100, 1, 1, 7.1999),
    (0.2, 100, 1, 3, 5.5107),
    (0.2, 100, 1, 8, 1.9836),
    (0.2, 104, 1, 1, 9.4775),
    (0.2, 104, 1, 3, 7.714),
    (0.2, 104, 1, 8, 3.7678),
]

std_american_put = [
    # S, r, d, expected
    (95, 0.02, 0.00, 9.559604714303656),
    (105, 0.02, 0.00, 5.182835507452405),
    (95, 0.05, 0.00, 8.452646381525147),
    (105, 0.05, 0.00, 4.306098201373055),
    (95, 0.1, 0.00, 7.138640520662231),
    (105, 0.1, 0.00, 3.2045597568873134),
    # d = 0.04
    (95, 0.02, 0.04, 11.385533383653373),
    (105, 0.02, 0.04, 6.5735670941797775),
    (95, 0.05, 0.04, 9.755163939182946),
    (105, 0.05, 0.04, 5.362941132857127),
    (95, 0.1, 0.04, 8.011765981901965),
    (105, 0.1, 0.04, 3.9592945293385684),
]