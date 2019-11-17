'''
Time Series Analysis: Homework II
Austin Koenig
'''

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

sys.stdout = open('./log.txt', 'w')

# //////////////////////////////// Problem 1 ////////////////////////////////

print("Problem 1: Assume the `robberies.csv` dataset.")
robberies = pd.read_csv('./datasets/robberies.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
robberies_plot = robberies.plot(x = 0, y = 1)
robberies_figure = robberies_plot.get_figure()
robberies_figure.suptitle('Robberies', fontsize=14)
robberies_figure.savefig('./figures/robberies.pdf')

print("  a. Perform a Dickey-Fuller test on the series. Is the series stationary?")
adf_result = adfuller(robberies.values)
print(f"\nADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print("Critical Values:")
for key, val in adf_result[4].items():
    print(f"[ {key} , {val} ]")


# //////////////////////////////// Problem 2 ////////////////////////////////



# //////////////////////////////// Problem 3 ////////////////////////////////



# //////////////////////////////// Problem 4 ////////////////////////////////