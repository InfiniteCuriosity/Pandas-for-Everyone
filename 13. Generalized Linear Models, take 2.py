# Generalized Linear Models

#13.2 Logistic Regression

'''When you have a binary response variable, logistic regression is often used to model the data.
Here’s some data from the American Community Survey (ACS) for New York.'''

##### Import #####
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from sklearn import linear_model
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

##################

acs = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/acs_ny.csv')
print(acs.head())
acs.columns
print(acs.columns) # the print function does not change the output
acs.info

'''To model these data, we first need to create a binary response variable.
Here we split the FamilyIncome variable into a binary variable.'''

acs['ge150k'] = pd.cut(acs['FamilyIncome'], [0, 150000, acs['FamilyIncome'].max()], labels=[0,1])
acs['ge150k_i'] = acs['ge150k'].astype(int)
print(acs['ge150k_i'].value_counts())
acs.info()

# 13.2.1 GLM using Statsmodels

model = smf.logit('ge150k_i ~ HouseCosts + NumWorkers + OwnRent + NumBedrooms + FamilyType', data = acs)
results = model.fit()
print(results.summary())

# To interpret our logistic model, we first need to exponentiate our results.

odds_ratios = np.exp(results.params)
print(odds_ratios)

#  An example interpretation of these numbers would be that for every one unit increase in NumBedrooms,
#  the odds of the FamilyIncome being greater than 150,000 increases by 1.27 times.

'''An example interpretation of these dummy variables would be that the odds of the FamilyIncome being greater than 150,000
increases by 1.82 times when the home is owned outright versus being under a mortgage.'''

# 13.2.2 Using Sklearn

# When using sklearn, remember that dummy variables need to be created manually. UGH!!!

predictors = pd.get_dummies(
    acs[['HouseCosts', 'NumWorkers', 'OwnRent', 'NumBedrooms', 'FamilyType']], drop_first=True
)

lr = linear_model.LogisticRegression()

# We can fit our model in the same way as when we fitted a linear regression.
results = lr.fit(X = predictors, y = acs['ge150k_i']) # this returns a warning but not an error

print(results.coef_) # this runs
print(results.intercept_) # this runs

# print results in a more attractive format

values = np.append(results.intercept_, results.coef_)
names = np.append('intercept', predictors.columns)

# put everything into a labeled data frame

results = pd.DataFrame(values, index = names, columns=['coef'])

# exponentiate our values for accurate interpretation

results['or'] = np.exp(results['coef'])
print(results)

# 13.3 Poisson Regression

# 13.3.1 Using Statsmodels

results = smf.poisson(
            'NumChildren ~ FamilyIncome + FamilyType + OwnRent',
            data=acs).fit() # this returns four warnings and does not converge

print(results.summary()) # does not converge, results are nan

model = smf.glm('NumChildren ~ FamilyIncome + FamilyType + OwnRent', data = acs,
                family = sm.families.Poisson(sm.genmod.families.links.log)) # warning this is deprecated

model = smf.glm('NumChildren ~ FamilyIncome + FamilyType + OwnRent', data = acs,
                family = sm.families.Poisson()) # this runs without a warning and returns the same results
results = model.fit()
print(results.summary())

# 13.3.2 Negative Binomial Regression for Overdispersion

# If our assumptions for Poisson regression are violated—that is, if our data has overdispersion—we can perform a negative binomial regression instead.

model = smf.glm('NumChildren ~ FamilyIncome + FamilyType + OwnRent',
                data=acs,
                family=sm.families.NegativeBinomial())
print(results.summary()) # this converges and runs successfully!

# 13.5 Survival Analysis does not work for me at all, and I cannot repair it to work

#  survival analysis is used when modeling the time to occurrence of a certain event

bladder = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/bladder.csv')
bladder.head(15)
# let's find the counts of the different treatments :)

print(bladder['rx'].value_counts()) # 1: 188, 2: 252

'''Creating the model and fitting the data proceed similarly to how models are fit using sklearn.
The stop variable indicates when an event occurs,
and the event variable signals whether the event of interest (bladder cancer re-occurrence) occurred. '''

kmf = KaplanMeierFitter()
kmf.fit(bladder['stop'], event_observed=bladder['event']) # returns a number of TypeErrors


#We can plot the survival curve using matplotlib - I cannot get this to run correctly! :(

fix, ax = plt.subplots()
ax = kmf.survival_function_.plot(ax = ax)
ax.set_title('Survival function of political regimes')
plt.show()
