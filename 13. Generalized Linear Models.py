# 13. Generalized Linear Models

# 13.2 Logistic Regression

'''When you have a binary response variable, logistic regression is often used to model the data.
Here’s some data from the American Community Survey (ACS) for New York.'''

import pandas as pd
acs = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/acs_ny.csv')
print(acs.columns)

print(acs.head())

'''To model these data, we first need to create a binary response variable.
Here we split the FamilyIncome variable into a binary variable.'''

acs['ge150k'] = pd.cut(acs['FamilyIncome'], [0, 150000, acs['FamilyIncome'].max()], labels = [0,1])
print(acs['ge150k'])

acs['ge150k_i'] = acs['ge150k'].astype(int) # counts how many 0s and 1s are in the column of data
print(acs['ge150k_i'].value_counts()) # 18294 0s and #4451 1s

# let's look at our binary 0/1 variable:

acs.info()

# 13.2.1 Using Statsmodels

import statsmodels.formula.api as smf

model = smf.logit('ge150k_i ~ HouseCosts + NumWorkers + OwnRent + NumBedrooms + FamilyType', data = acs)
results = model.fit()
print(results.summary())

import numpy as np

odds_raios = np.exp(results.params)
print(odds_raios)

# Using Sklearn

# when using Sklearn, recall that dummy variables need to be created manually

predictors = pd.get_dummies(acs[['HouseCosts', 'NumWorkers', 'OwnRent', 'NumBedrooms', 'FamilyType']], drop_first=True)

# We can then use the LogisticRegression object from the linear_model module.

from sklearn import linear_model
lr = linear_model.LogisticRegression()

results = lr.fit(X = predictors, y =acs['ge150k_i'])
print(results.coef_)

# print the intercept:

print(results.intercept_)

values = np.append(results.intercept_, results.coef_)
names = np.append('intercept', predictors.columns)

# assemble everything into a labeled data frame

results = pd.DataFrame(values, index = names, columns=['coef'])
print(results)

# In order to interpret our coefficients, we still need to exponentiate our values.

results['or'] = np.exp(results['coef'])
print(results)

# 13.3 Poisson Regression

# We can perform a Poisson regression using the poisson function in statsmodels.

results = smf.poisson('NumChildren ~ FamilyIncome + FamilyType + OwnRent', data = acs).fit()
print(results.summary)

'''The benefit of using a generalized linear model is that the only things that need to be changed are
the family of the model that needs to be fit, and the link function that transforms our data.
We can also use the more general glm function to perform all the same calculations.'''

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.glm('NumChildren ~ FamilyIncome + FamilyType + OwnRent', data=acs, family = sm.families.Poisson(sm.genmod.families.links.log))
results = model.fit()
print(results.summary())

# 13.3.2 Negative Binomial Regression for Overdispersion

'''If our assumptions for Poisson regression are violated—that is, if our data has overdispersion—
we can perform a negative binomial regression instead.'''

model = smf.glm('NumChildren ~ FamilyIncome + FamilyType + OwnRent', data = acs, family = sm.families.NegativeBinomial(sm.genmod.families.links.log))
results = model.fit()
print(results.summary())

#13.4 More Generalized Linear Models

'''The documentation page for GLM found in statsmodels1 lists the various families that can be passed into the glm parameter. These families can all be found under sm.families.<FAMILY>:
. Binomial
. Gamma
. InverseGaussian . NegativeBinomial . Poisson
. Tweedie
The link functions are found under sm.families.family.<FAMILY>.links. Following is the list of link functions, but note that not all link functions are available for each family:
. CDFLink
. CLogLog
. Log
. Logit
. NegativeBinomial . Power
. cauchy
. identity
. inverse_power
. inverse_squared'''

# 13.5 Survival Analysis

bladder = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/bladder.csv')
print(bladder.head())

# Here are the counts of the different treatments, rx.
print(bladder['rx'].value_counts())

from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(bladder['stop'], event_observed = bladder['event'])

# We can plot the survival curve using matplotlib, as shown in Figure 13.1.

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax = kmf.survival_function_.plot(ax = ax)
ax.set_title('Survival function of political regimes')
plt.show()

'''So far we’ve just plotted the survival curve. We can also fit a model
to predict survival rate. One such model is called the Cox proportional hazards model.
We fit this model using the CoxPHFitter class from lifelines.'''

from lifelines import CoxPHFitter
cph = CoxPHFitter()
# We then pass in the columns to be used as predictors.
cph_bladder_df = bladder[['rx', 'number', 'size', 'enum', 'stop', 'event']]
cph.fit(cph_bladder_df, duration_col = 'stop', event_col = 'event')
print(cph.print_summary())

# 13.5.1 Testing the Cox Model Assumptions

'''One way to check the Cox model assumptions is to plot a separate survival curve
by strata. In our example, our strata will be the values of the rx column, meaning we will
plot a separate curve for each type of treatment.
If the log(-log(survival curve)) versus log(time) curves cross each other (Figure 13.3),
then it signals that the model needs to be stratified by the variable.'''

rx1 = bladder.loc[bladder['rx'] == 1]
rx2 = bladder.loc[bladder['rx'] == 2]

kmf1 = KaplanMeierFitter()
kmf1.fit(rx1['stop'], event_observed = rx1['event'])

kmf2 = KaplanMeierFitter()
kmf2.fit(rx2['stop'], event_observed = rx2['event'])

fit, axes = plt.subplots()

# put both plots on the same axes

kmf1.plot_loglogs(ax = axes)
kmf2.plot_loglogs(ax = axes)
axes.legend(['rx1', 'rx2'])

plt.show()

cph_strat = CoxPHFitter()
cph_strat.fit(cph_bladder_df, duration_col = 'stop', event_col = 'event', strata = ['rx'])
print(cph_strat.print_summary())

'''This chapter covered some of the most basic and common models used in data analysis.
These types of models serve as an interpretable baseline for more complex machine learning models.
As we cover more complex models, keep in mind that sometimes simple and tried-and-true
interpretable models can outperform the fancy newer models.'''