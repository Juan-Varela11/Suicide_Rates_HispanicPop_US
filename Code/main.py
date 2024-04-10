import pandas as pd
import numpy as np
import sklearn 
from math import *
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import optimize
import sklearn.svm

# Load in data
yr1985to2018_data = pd.read_csv('Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States_20240327.csv')
yr2018to2023_Ages_data = pd.read_csv('ProvisionalMortalityStatistics_2018_2024_Hispanic_AgeBrackets_Revised.csv')
yr2018to2023_Gender_data = pd.read_csv('ProvisionalMortalityStatistics_2018_2024_Hispanic_Genders_Revised.csv')

# Drop unneeded features from datasets 
yr2018to2023_Ages_data = yr2018to2023_Ages_data.drop(columns=['Notes','Gender Code','Ten-Year Age Groups','Year'])
yr2018to2023_Gender_data = yr2018to2023_Gender_data.drop(columns=['Notes','Gender Code','Year'])

# Isolate crude data (unadjusted rates) for Male Hispanic or Latino AND Female Hispanic or Latino instances (regardless of race)
# Data to plot year vs. crude estimate (all ages - male)
male_hispanic_cdc = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Male: Hispanic or Latino: All races') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
male_hispanic_cdc = male_hispanic_cdc.dropna(subset=['ESTIMATE'])
male_hispanic_cdc_2018to2023 = yr2018to2023_Gender_data[yr2018to2023_Gender_data['Gender']== 'Male']

# Data to plot year vs. crude estimate (different age brackets - male)
male_hispanic_cdc_15to24 = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Male: Hispanic or Latino: All races: 15-24 years') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
male_hispanic_cdc_15to24 = male_hispanic_cdc_15to24.dropna(subset=['ESTIMATE'])
male_hispanic_cdc_25to44 = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Male: Hispanic or Latino: All races: 25-44 years') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
male_hispanic_cdc_25to44 = male_hispanic_cdc_25to44.dropna(subset=['ESTIMATE'])
male_hispanic_cdc_45to64 = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Male: Hispanic or Latino: All races: 45-64 years') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
male_hispanic_cdc_45to64 = male_hispanic_cdc_45to64.dropna(subset=['ESTIMATE'])
male_hispanic_cdc_65above = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Male: Hispanic or Latino: All races: 65 years and over') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
male_hispanic_cdc_65above = male_hispanic_cdc_65above.dropna(subset=['ESTIMATE'])

# Data to plot year vs. crude estimate (all ages - female)
female_hispanic_cdc = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Female: Hispanic or Latino: All races') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
female_hispanic_cdc = female_hispanic_cdc.dropna(subset=['ESTIMATE'])
female_hispanic_cdc_2018to2023 = yr2018to2023_Gender_data[yr2018to2023_Gender_data['Gender']== 'Female']

# Data to plot year vs. crude estimate (different age brackets - female)
female_hispanic_cdc_15to24 = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Female: Hispanic or Latino: All races: 15-24 years') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
female_hispanic_cdc_15to24 = female_hispanic_cdc_15to24.dropna(subset=['ESTIMATE'])
female_hispanic_cdc_25to44 = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Female: Hispanic or Latino: All races: 25-44 years') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
female_hispanic_cdc_25to44 = female_hispanic_cdc_25to44.dropna(subset=['ESTIMATE'])
female_hispanic_cdc_45to64 = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Female: Hispanic or Latino: All races: 45-64 years') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
female_hispanic_cdc_45to64 = female_hispanic_cdc_45to64.dropna(subset=['ESTIMATE'])
female_hispanic_cdc_65above = yr1985to2018_data[(yr1985to2018_data['STUB_LABEL'] == 'Female: Hispanic or Latino: All races: 65 years and over') & (yr1985to2018_data['UNIT'] == 'Deaths per 100,000 resident population, crude')]
female_hispanic_cdc_65above = female_hispanic_cdc_65above.dropna(subset=['ESTIMATE'])


# Concatenating datasets for more complete analysis and visualizations 
gender_cdc_data = pd.concat([male_hispanic_cdc,female_hispanic_cdc])
age_male_cdc_data = pd.concat([male_hispanic_cdc_15to24,male_hispanic_cdc_25to44,
                                 male_hispanic_cdc_45to64,male_hispanic_cdc_65above])
age_female_cdc_data = pd.concat([female_hispanic_cdc_15to24,female_hispanic_cdc_25to44,
                                 female_hispanic_cdc_45to64,female_hispanic_cdc_65above])

# Visualize the relationship between year and number of suicide deaths (crude estimate) for the different genders
plt.scatter(male_hispanic_cdc['YEAR'],male_hispanic_cdc['ESTIMATE'],label='Male',c='blue')
plt.scatter(male_hispanic_cdc_2018to2023['Year Code'],male_hispanic_cdc_2018to2023['Crude Rate'],c='blue')
plt.scatter(female_hispanic_cdc['YEAR'],female_hispanic_cdc['ESTIMATE'],label='Female',c='red')
plt.scatter(female_hispanic_cdc_2018to2023['Year Code'],female_hispanic_cdc_2018to2023['Crude Rate'],c='red')
plt.title('Suicide Death Rates in US Hispanic & Latino Communities per Year')
plt.xlabel('Year')
plt.ylabel('Suicide Deaths (per 100,000 residents)')
plt.legend(loc='best')
plt.savefig(fname='deaths_vs_year.jpg')
plt.close()

#fig1 = px.scatter(gender_cdc_data,x='YEAR',y='ESTIMATE',color='STUB_LABEL',
#                  labels={
#                      'YEAR': 'Year',
#                      'ESTIMATE': 'Suicide Deaths (per 100,000 residents)',
#                      'STUB_LABEL': 'Genders'
#                  },
#                  title="Suicide Death Rates in US Hispanic & Latino Communities per Year")
#fig1.write_html("deaths_vs_year.html")


# Visualize the relationship between year and number of suicide deaths (crude estimate) for the different ages (male)
plt.scatter(male_hispanic_cdc_15to24['YEAR'],male_hispanic_cdc_15to24['ESTIMATE'],label='15-24 years',c='blue')
plt.scatter(male_hispanic_cdc_25to44['YEAR'],male_hispanic_cdc_25to44['ESTIMATE'],label='25-44 years',c='red')
plt.scatter(male_hispanic_cdc_45to64['YEAR'],male_hispanic_cdc_45to64['ESTIMATE'],label='45-64 years',c='green')
plt.scatter(male_hispanic_cdc_65above['YEAR'],male_hispanic_cdc_65above['ESTIMATE'],label='65 years and above',c='orange')
plt.title('Suicide Death Rates in US Hispanic & Latino Male Communities per Year')
plt.xlabel('Year')
plt.ylabel('Suicide Deaths (per 100,000 residents)')
plt.legend(loc='best')
plt.savefig(fname='deaths_vs_year_male_ages.jpg')
plt.close()

#fig2 = px.scatter(age_male_cdc_data,x='YEAR',y='ESTIMATE',color='STUB_LABEL',
#                  labels={
#                      'YEAR': 'Year',
#                      'ESTIMATE': 'Suicide Deaths (per 100,000 residents)',
#                      'STUB_LABEL': 'Age Brackets'
#                  },
#                  title="Suicide Death Rates in US Hispanic & Latino Male Communities per Year")
#fig2.write_html("deaths_vs_year_male_ages.html")

# Visualize the relationship between year and number of suicide deaths (crude estimate) for the different ages (female)
plt.scatter(female_hispanic_cdc_15to24['YEAR'],female_hispanic_cdc_15to24['ESTIMATE'],label='15-24 years',c='blue')
plt.scatter(female_hispanic_cdc_25to44['YEAR'],female_hispanic_cdc_25to44['ESTIMATE'],label='25-44 years',c='red')
plt.scatter(female_hispanic_cdc_45to64['YEAR'],female_hispanic_cdc_45to64['ESTIMATE'],label='45-64 years',c='green')
plt.scatter(female_hispanic_cdc_65above['YEAR'],female_hispanic_cdc_65above['ESTIMATE'],label='65 years and above',c='orange')
plt.title('Suicide Death Rates in US Hispanic/Latino Female Communities per Year')
plt.xlabel('Year')
plt.ylabel('Suicide Deaths (per 100,000 residents)')
plt.legend(loc='best')
plt.savefig(fname='deaths_vs_year_female_ages.jpg')
plt.close()

#fig3 = px.scatter(age_female_cdc_data,x='YEAR',y='ESTIMATE',color='STUB_LABEL',
#                  labels={
#                      'YEAR': 'Year',
#                      'ESTIMATE': 'Suicide Deaths (per 100,000 residents)',
#                      'STUB_LABEL': 'Age Brackets'
#                  },
#                  title="Suicide Death Rates in US Hispanic & Latino Female Communities per Year")
#fig3.write_html("deaths_vs_year_female_ages.html")

# Determine a best-fit polynomial for year vs crude estimate
def cdc_fit(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d 

params_male, params_covariance_male = optimize.curve_fit(cdc_fit, male_hispanic_cdc['YEAR'], male_hispanic_cdc['ESTIMATE'])
params_female, params_covariance_female = optimize.curve_fit(cdc_fit, female_hispanic_cdc['YEAR'], female_hispanic_cdc['ESTIMATE'])

# Plotting polynomial of best fit to year vs crude estimate for different genders for years 1985 - 2018
# Observing how estimates for 2018 - 2023 follow the fit 
plt.scatter(male_hispanic_cdc['YEAR'],male_hispanic_cdc['ESTIMATE'],label='Male',c='blue')
plt.scatter(female_hispanic_cdc['YEAR'],female_hispanic_cdc['ESTIMATE'],label='Female',c='red')
plt.plot(male_hispanic_cdc['YEAR'], cdc_fit(male_hispanic_cdc['YEAR'],params_male[0],
                                                 params_male[1],params_male[2],
                                                 params_male[3]),
                                                 label='Male data fit',c='blue')
plt.plot(female_hispanic_cdc['YEAR'], cdc_fit(female_hispanic_cdc['YEAR'],params_female[0],
                                                 params_female[1],params_female[2],
                                                 params_female[3]),
                                                 label='Female data fit',c='red')
plt.title('Suicide Death Rates in US Hispanic/Latino Community per Year')
plt.xlabel('Year')
plt.ylabel('Suicide Death Rate (per 100,000 residents)')
plt.legend(loc='best')
plt.savefig(fname='deaths_vs_year_withfit.jpg')
plt.close()

# Extending the poly by 10 years, to see how the fit evolves for the different genders
time_future = pd.Series([2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028])

plt.scatter(male_hispanic_cdc['YEAR'],male_hispanic_cdc['ESTIMATE'],label='Male',c='blue')
plt.scatter(female_hispanic_cdc['YEAR'],female_hispanic_cdc['ESTIMATE'],label='Female',c='red')
plt.scatter(male_hispanic_cdc_2018to2023['Year Code'],male_hispanic_cdc_2018to2023['Crude Rate'],
            c='blue',marker='x')
plt.scatter(female_hispanic_cdc_2018to2023['Year Code'],female_hispanic_cdc_2018to2023['Crude Rate'],
            c='red',marker='x')
plt.plot(male_hispanic_cdc['YEAR'], cdc_fit(male_hispanic_cdc['YEAR'],params_male[0],
                                                 params_male[1],params_male[2],
                                                 params_male[3]),
                                                 c='blue')
plt.plot(female_hispanic_cdc['YEAR'], cdc_fit(female_hispanic_cdc['YEAR'],params_female[0],
                                                 params_female[1],params_female[2],
                                                 params_female[3]),
                                                 c='red')
plt.plot(time_future, cdc_fit(time_future,params_male[0],
                                                 params_male[1],params_male[2],
                                                 params_male[3]),
                                                 c='blue',linestyle = 'dashed')
plt.plot(time_future, cdc_fit(time_future,params_female[0],
                                                 params_female[1],params_female[2],
                                                 params_female[3]),
                                                 c='red',linestyle = 'dashed')
plt.title('Suicide Death Rates vs. Year - Extended Fit')
plt.xlabel('Year')
plt.ylabel('Suicide Death Rate (per 100,000 residents)')
plt.legend(loc='best')
plt.savefig(fname='deaths_vs_year_fitextended.jpg')
plt.close()

# Use sklearn to predict future values and compare how they line up to current scipy fit (W.I.P.)

# Creating feature matrix and target vector 
features_names = []
x_male = male_hispanic_cdc.loc[:,features_names].values
x_female = female_hispanic_cdc.loc[:,features_names].values
