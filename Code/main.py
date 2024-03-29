import pandas as pd
import numpy as np
import sklearn as sk
from math import *
import plotly
import matplotlib.pyplot as plt

# Load in data
cdc_data = pd.read_csv('Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States_20240327.csv')

# Explore & describe data
print(cdc_data.describe())
print(cdc_data.shape)
print(cdc_data.head(5))
print(cdc_data.info(verbose=True))

# Isolate data for Male Hispanic or Latino AND Female Hispanic or Latino instances
