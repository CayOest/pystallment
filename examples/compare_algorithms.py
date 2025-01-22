import numpy as np
import pandas as pd
from itertools import product
import pystallment.option as opt
import pystallment.algorithms.fdm as fdm
import pystallment.algorithms.lsmc as lsmc
import pystallment.algorithms.binomial as bin

def generate_test_data(S, K, r, d, vola, T, q):
    data = [S, K, r, d, vola, T, q]
    return product(*data)

if __name__ == "__main__":
    print("Compare Algorithms")
    S = [95, 100, 105]
    K = [100]
    r = [0.02, 0.05]
    d = [0.0, 0.02]
    vola = [0.2]
    T = [1.0]
    q = [3, 5, 8]
    data = generate_test_data(S, K, r, d, vola, T, q)

    results = []

    for (S, K, r, d, vola, T, q) in data:
        option = opt.ContinuousInstallmentOption(S, K, r, d, vola, T, q, +1)
        print(repr(option))

        fdm_pricer = fdm.FDMPricer(option)
        fdm_price = fdm_pricer.price()

        lsmc_prices = []
        for i in range(100):
            lsmc_pricer = lsmc.LSMCPricer(option)
            lsmc_pricer.num_paths = 100000
            lsmc_prices.append( lsmc_pricer.price() )

        lsmc_price = np.mean(lsmc_prices)

        bin_pricer = bin.BinomialPricer(option)
        bin_price = bin_pricer.price()

        results.append({
            "S": S,
            "K": K,
            "r": r,
            "d": d,
            "vola": vola,
            "T": T,
            "q": q,
            "FDM": round(fdm_price, 4),
            "Binomial": round(bin_price, 4),
            "LSMC": round(lsmc_price, 4),
            "FDM-Binomial Diff": (fdm_price - bin_price),
            "FDM-LSMC Diff": (fdm_price-lsmc_price),
            "Binomial-LSMC Diff": (bin_price - lsmc_price),
        })

    # Ergebnisse in eine Pandas-Tabelle umwandeln
    df = pd.DataFrame(results)
    print(df)

    df.to_csv("algorithm_comparison.csv", index=False)

    # Optional: Tabelle mit schönem Stil ausgeben (für Jupyter Notebooks)
    try:
        from IPython.display import display
        display(df.style.format({"FDM Price": "{:.4f}",
                                 "LSMC Price": "{:.4f}",
                                 "Binomial Price": "{:.4f}"}))
    except ImportError:
        pass
