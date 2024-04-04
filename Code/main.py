import pandas as pd
import numpy as np
import sklearn as sk
from math import *
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import optimize

# Load in data
cdc_data = pd.read_csv('Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States_20240327.csv')

# Explore & describe complete cdc dataset
print(cdc_data.describe())
print(cdc_data.shape)
print(cdc_data.head(1))
print(cdc_data.columns.values)
print(cdc_data.info(verbose=True))

# Isolate crude data (unadjusted rates) for Male Hispanic or Latino AND Female Hispanic or Latino instances (regardless of race)
male_hispanic_cdc = cdc_data[(cdc_data['STUB_LABEL'] == 'Male: Hispanic or Latino: All races') & (cdc_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
male_hispanic_cdc = male_hispanic_cdc.dropna(subset=['ESTIMATE'])
female_hispanic_cdc = cdc_data[(cdc_data['STUB_LABEL'] == 'Female: Hispanic or Latino: All races') & (cdc_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
female_hispanic_cdc = female_hispanic_cdc.dropna(subset=['ESTIMATE'])

male_hispanic_cdc.to_csv('male_hispanic_cdc.csv')
female_hispanic_cdc.to_csv('female_hispanic_cdc.csv')  

# Visualize the relationship between year and number of suicide deaths
plt.scatter(male_hispanic_cdc['YEAR'],male_hispanic_cdc['ESTIMATE'],label='Male',c='blue')
plt.scatter(female_hispanic_cdc['YEAR'],female_hispanic_cdc['ESTIMATE'],label='Female',c='red')
plt.title('Estimate of Suicide Deaths in US Hispanic/Latino Community per Year')
plt.xlabel('Year')
plt.ylabel('# of Suicide Deaths')
plt.legend(loc='best')
plt.savefig(fname='deaths_vs_year.jpg')
plt.close()

# Determine a best-fit polynomial for the plots
def male_cdc_fit(x,a,b,c,d,e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e 

def female_cdc_fit(x,a,b,c,d,e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e 

params_male, params_covariance_male = optimize.curve_fit(male_cdc_fit, male_hispanic_cdc['YEAR'], male_hispanic_cdc['ESTIMATE'])
params_female, params_covariance_female = optimize.curve_fit(female_cdc_fit, female_hispanic_cdc['YEAR'], female_hispanic_cdc['ESTIMATE'])

# Plotting polynomial of best fit to the data
plt.scatter(male_hispanic_cdc['YEAR'],male_hispanic_cdc['ESTIMATE'],label='Male',c='blue')
plt.scatter(female_hispanic_cdc['YEAR'],female_hispanic_cdc['ESTIMATE'],label='Female',c='red')
plt.plot(male_hispanic_cdc['YEAR'], male_cdc_fit(male_hispanic_cdc['YEAR'],params_male[0],
                                                 params_male[1],params_male[2],
                                                 params_male[3],params_male[4]),
                                                 label='Male data fit',c='blue')
plt.plot(female_hispanic_cdc['YEAR'], female_cdc_fit(female_hispanic_cdc['YEAR'],params_female[0],
                                                 params_female[1],params_female[2],
                                                 params_female[3],params_female[4]),
                                                 label='Female data fit',c='red')
plt.title('Estimate of Suicide Deaths in US Hispanic/Latino Community per Year')
plt.xlabel('Year')
plt.ylabel('# of Suicide Deaths')
plt.legend(loc='best')
plt.savefig(fname='deaths_vs_year_withfit.jpg')
plt.close()
