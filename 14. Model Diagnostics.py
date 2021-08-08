# 14.1  Introduction

'''Building models is a continuous art. As we start adding and removing variables from our models,
we need a means to compare models with one another and a consistent way of measuring model
performance. There are many ways we can compare models,
and this chapter describes some of these methods.'''

#14.2 Residuals

'''The residuals of a model compare what the model calculates and the actual values in the data.
Let’s fit some models on a housing data set.'''

import pandas as pd
pd.set_option('display.max_columns', None)
housing = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/housing_renamed.csv')
print(housing.head())

# We’ll begin with a multiple linear regression model with three covariates.

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import statsmodels.regression.linear_model

house1 = smf.glm('value_per_sq_ft ~ units + sq_ft + boro', data = housing).fit()
print(house1.summary())

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax = sns.regplot(x = house1.fittedvalues, y = house1.resid_deviance, fit_reg = False)
plt.show()

# color the points by Boro

res_df = pd.DataFrame({
    'fittedvalues': house1.fittedvalues,
    'resid_deviance': house1.resid_deviance,
    'boro': housing['boro']
})

fig = sns.lmplot(x = 'fittedvalues', y ='resid_deviance', data = res_df, hue = 'boro',
                 fit_reg = False)

plt.show()
fig.savefit('p5-ch-model_diagnostics/figures/resid_boros')

# 14.2.1 Q-Q Plots

from scipy import stats
resid = house1.resid_deviance.copy()
resid_std = stats.zscore(resid)

fig = statsmodels.graphics.gofplots.qqplot(resid, line = 'r')
plt.show() # looks good for the middle, the bottom is off, the top ~15% is way off the line

# plot a histogram of the residuals to see if our data is normally distributed

fig, ax = plt.subplots()
ax = sns.distplot(resid_std)
plt.show()

# 14.3 Comparing Multiple Models

f1 = 'value_per_sq_ft ~ units + sq_ft + boro'
f2 = 'value_per_sq_ft ~ units * sq_ft + boro'
f3 = 'value_per_sq_ft ~ units + sq_ft * boro + type'
f4 = 'value_per_sq_ft ~ units + sq_ft + boro + sq_ft * type'
f5 = 'value_per_sq_ft ~ boro + type'

house1 = smf.ols(f1, data = housing).fit()
house2 = smf.ols(f2, data = housing).fit()
house3 = smf.ols(f3, data = housing).fit()
house4 = smf.ols(f4, data = housing).fit()
house5 = smf.ols(f5, data = housing).fit()

'''
With all our models, we can collect all of our coefficients
and the model with which they are associated.'''

mod_results = pd.concat([house1.params, house2.params, house3.params,
                         house4.params, house5.params], axis = 1).rename(columns = lambda x: 'house' + str(x+1)).reset_index().rename(columns={'index' : 'param'}).melt(id_vars='param', var_name = 'model', value_name='estimate')

print(mod_results.head())
print(mod_results.tail())

'''we can plot our coefficients to quickly see how the models are estimating parameters in relation to each other (Figure 14.5).'''

fig, ax = plt.subplots()
ax = sns.pointplot(x = "estimate", y = "param", hue = "model",
                   data = mod_results,
                   dodge = True,
                   join = False)

plt.tight_layout()
plt.show()

'''Now that we have our linear models, we can use the analysis of variance (ANOVA) method to compare them. The ANOVA will give us the residual sum of sqaures (RSS),
which is one way we can measure performance (lower is better).'''

model_names = ['house1', 'house2', 'house3', 'house4', 'house5']
house_anova = statsmodels.stats.anova.anova_lm(house1, house2, house3, house4,house5)
house_anova.index = model_names
print(house_anova)

'''Another way we can calculate model performance is by using the Akaike information criterion (AIC)
and the Bayesian information criterion (BIC). These methods apply a penalty for each feature that is added to the model.
Thus, we should strive to balance performance and parsimony (lower is better).'''

house_models = [house1, house2, house3, house4, house5]

house_aic = list(map(statsmodels.regression.linear_model.RegressionResults.aic, house_models))
house_bic = list(map(statsmodels.regression.linear_model.RegressionResults.bic, house_models))

house_aic = list(
     map(statsmodels.regression.linear_model.RegressionResults.aic,
         house_models))


abic = pd.DataFrame({
    'model' : model_names,
    'aic' : house_aic,
    'bic' : house_bic
})

print(abic)

# 14.3.2 Working with GLM Models

'''We can perform the same calculations and model diagnostics on
generalized linear models (GLMs).
However, the ANOVA is simply the deviance of the model.'''

def anova_devianace_table(*models):
    return pd.DataFrame({
        'df_residuals': [i.df_resid for i in models],
        'resid_stddev': [i.devinacne for i in models],
        'df': [i.df_model for i in models],
        'deviance': [i.deviance for i in models]
    })
f1 = 'value_per_sq_ft ~ units + sq_ft + boro'
f2 = 'value_per_sq_ft ~ units * sq_ft + boro'
f3 = 'value_per_sq_ft ~ units + sq_ft * boro + type'
f4 = 'value_per_sq_ft ~ units + sq_ft * boro + sq_ft * type'
f5 = 'value_per_sq_ft ~ boro + type'

glm1 = smf.glm(f1, data = housing).fit()
glm2 = smf.glm(f2, data = housing).fit()
glm3 = smf.glm(f3, data = housing).fit()
glm4 = smf.glm(f4, data = housing).fit()
glm5 = smf.glm(f5, data = housing).fit()
glm_anova = anova_deviance_table(glm1, glm2, glm3, glm4, glm5)
print(glm_anova) # that worked!

# we can do the same set of calculations in a logistic regression

# create a binary variable
housing['high_value'] = (housing['value_per_sq_ft'] >= 150).astype(int)

print(housing['high_value'].value_counts())

# create and fit our logistic regression using GLM

f1 = 'high_value ~ units + sq_ft + boro'
f2 = 'high_value ~ units * sq_ft + boro'
f3 = 'high_value ~ units + sq_ft * boro + type'
f4 = 'high_value ~ units + sq_ft * boro + sq_ft * type'
f5 = 'high_value ~ boro + type'

logistic = statsmodels.genmod.families.family.Binomial() # error correction at https://stackoverflow.com/questions/59655369/how-to-run-a-glm-gamma-regression-in-python-with-r-like-formulas

glm1 = smf.glm(f1, data = housing, family = logistic).fit()
glm2 = smf.glm(f2, data = housing, family = logistic).fit()
glm3 = smf.glm(f3, data = housing, family = logistic).fit()
glm4 = smf.glm(f4, data = housing, family = logistic).fit()
glm5 = smf.glm(f5, data = housing, family = logistic).fit()

# show the deviances from our GLM models

print(anova_deviance_table(glm1, glm2, glm3, glm4, glm5)) # not sure why PyCharm is givng an error here, because this runs

mods = [glm1, glm2, glm3, glm4, glm5]

mods_aic = list(
    map(statsmodels.regression.linear_model.RegressionResults.aic, mods)
)

mods_bic = list(
    map(statsmodels.regression.linear_model.RegressionResults.bic, mods)
)

# Remember, dicts are unordered
abic = pd.DataFrame({
    'model': model_names,
    'aic': house_aic,
    'bic': house_bic
})

print(abic) # this is not working for me!

# 14.4 k-Fold Cross Validation

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print(housing.columns)

# get the training and test data

X_train, X_test, y_train, y_test = train_test_split(
    pd.get_dummies(housing[['units', 'sq_ft', 'boro']], drop_first=True),
    housing['value_per_sq_ft'],
    test_size=0.20,
    random_state=42
)

# We can get a score that indicates how well our model is performing using our test data.

lr = LinearRegression().fit(X_train, y_train)
print(lr.score(X_test, y_test))

# To perform a k-fold cross-validation, we need to import this function from sklearn.

from sklearn.model_selection import KFold, cross_val_score
from patsy import dmatrices


# get a fresh new housing data set

housing = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/housing.csv')

# specify how many folds we want

kf = KFold(n_splits=5)

y, X = dmatrices('value_per_sq_ft ~ units + sq_ft + boro', housing)

# next we can train and test our model on each fold

coefs = []
scores = []
for train, test in kf.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    lr = LinearRegression().fit(X_train, y_train)
    coefs.append(pd.DataFrame(lr.coef_))
    scores.append(lr.score(X_test, y_test))

# view the results

coefs_df = pd.concat(coefs)
coefs_df.columns = X.design_info.column_names
coefs_df
