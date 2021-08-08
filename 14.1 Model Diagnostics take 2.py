# 14. Model Diagnostics

##### Import #####
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from patsy import dmatrices, build_design_matrices
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

##################

housing = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/housing_renamed.csv')
housing.head()
housing.info
housing.columns

house1 = smf.glm('value_per_sq_ft ~ units + sq_ft + boro', data = housing).fit()
print(house1.summary())

# We can plot the residuals of our model
fix, ax = plt.subplots()
ax = sns.regplot(x = house1.fittedvalues, y = house1.resid_deviance, fit_reg = False)
plt.show() #not that good, not able to discern many patterns, other than two large groups

# Let’s can color our plot by the boro variable

res_df = pd.DataFrame({
    'fittedvalues': house1.fittedvalues,
    'resid_deviance': house1.resid_deviance
    'boro': housing['boro']
})

fig = sns.lmplot(x = 'fittedvalues', y = 'resid_deviance', data = res_df, hue = 'boro', fit_reg=False)
plt.show()

# When we color our points based on boro, you can see that the clusters are highly governed by the value of this variable.

# 14.2.1 Q-Q Plots

resid = house1.resid_deviance.copy()
resid_std = stats.zscore(resid)
fig = statsmodels.graphics.gofplots.qqplot(resid, line = 'r')
plt.show()

# histogram of our residuals to see if our residuals are normally distributed

fig, ax = plt.subplots()
ax = sns.distplot(resid_std)
plt.show()

# 14.3 Compairng Multiple Models

# 14.3.1 Working with Linear Models

f1 = 'value_per_sq_ft ~ units + sq_ft + boro'
f2 = 'value_per_sq_ft ~ units * sq_ft + boro'
f3 = 'value_per_sq_ft ~ units + sq_ft * boro + type'
f4 = 'value_per_sq_ft ~ units + sq_ft * boro + sq_ft * type'
f5 = 'value_per_sq_ft ~ boro + type'
house1 = smf.ols(f1, data=housing).fit()
house2 = smf.ols(f2, data=housing).fit()
house3 = smf.ols(f3, data=housing).fit()
house4 = smf.ols(f4, data=housing).fit()
house5 = smf.ols(f5, data=housing).fit()

'''With all our models, we can collect all of our coefficients
and the model with which they are associated.'''

mod_results = pd.concat([house1.params, house2.params, house3.params, house4.params, house5.params], axis=1).rename(
    columns=lambda x: 'house' + str(x+1)).reset_index().rename(columns = {'index': 'param'}).melt(
    id_vars='param', value_name='model', var_name='estimate')
print(mod_results.head())
print(mod_results.tail())

#we can plot our coefficients to quickly see how the models are estimating parameters in relation to each other

fix, ax = plt.subplots()
ax = sns.pointplot(x = "estimate", y = "param", hue = "model",
                   data = mod_results,
                   dodge = True, # jitter the points
                    join = False) # don't connect the points

ax = sns.pointplot(x="estimate", y="param", hue="model", data=mod_results, dodge=True, # jitter the points
                   join=False) # don't connect the points

plt.tight_layout()
plt.show()

'''Now that we have our linear models, we can use the analysis of variance (ANOVA) method to compare them.
The ANOVA will give us the residual sum of sqaures (RSS),
which is one way we can measure performance (lower is better).'''

model_names = ['house1', 'house2', 'house3', 'house4', 'house5']
house_anova = statsmodels.stats.anova.anova_lm(house1, house2, house3, house4, house5)
house_anova.index = model_names
print(house_anova) # lowest SSR is house4

# 14.3.2 Working with GLM Models

'''We can perform the same calculations and model diagnostics on generalized linear models (GLMs).
However, the ANOVA is simply the deviance of the model.'''

def anova_deviance_table(*models):
    return pd.DataFrame({
        'df_residuals': [i.df_resid for i in models],
        'resid_stdev': [i.deviance for i in models],
        'df': [i.df_model for i in models],
        'deviance': [i.deviance for i in models]
    })

f1 = 'value_per_sq_ft ~ units + sq_ft + boro'
f2 = 'value_per_sq_ft ~ units * sq_ft + boro'
f3 = 'value_per_sq_ft ~ units + sq_ft * boro + type'
f4 = 'value_per_sq_ft ~ units + sq_ft * boro + sq_ft * type'
f5 = 'value_per_sq_ft ~ boro + type'
glm1 = smf.glm(f1, data=housing).fit()
glm2 = smf.glm(f2, data=housing).fit()
glm3 = smf.glm(f3, data=housing).fit()
glm4 = smf.glm(f4, data=housing).fit()
glm5 = smf.glm(f5, data=housing).fit()

glm_anova = anova_deviance_table(glm1, glm2, glm3,glm4, glm5)
print(glm_anova) # house2 has the lowest deviance

# We can do the same set of calculations in a logistic regression.

# Create a binary variable

housing['high_value'] = (housing['value_per_sq_ft']>=150).astype(int)
print(housing['high_value'].value_counts())

# create and fit our logistic regression using GLM

f1 = 'value_per_sq_ft ~ units + sq_ft + boro'
f2 = 'value_per_sq_ft ~ units * sq_ft + boro'
f3 = 'value_per_sq_ft ~ units + sq_ft * boro + type'
f4 = 'value_per_sq_ft ~ units + sq_ft * boro + sq_ft * type'
f5 = 'value_per_sq_ft ~ boro + type'

logistic = statsmodels.genmod.families.family.Binomial(link=statsmodels.genmod.families.links.logit)

glm1 = smf.glm(f1, data=housing, family=logistic).fit()
glm2 = smf.glm(f2, data=housing, family=logistic).fit()
glm3 = smf.glm(f3, data=housing, family=logistic).fit()
glm4 = smf.glm(f4, data=housing, family=logistic).fit()
glm5 = smf.glm(f5, data=housing, family=logistic).fit()

# show the deviance from our GLM models
print(anova_deviance_table(glm1, glm2, glm3, glm4, glm5))


##########

def anova_deviance_table(*models): return pd.DataFrame({
'df_residuals': [i.df_resid for i in models], 'resid_stddev': [i.deviance for i in models], 'df': [i.df_model for i in models], 'deviance': [i.deviance for i in models]
})

f1 = 'value_per_sq_ft ~ units + sq_ft + boro'
f2 = 'value_per_sq_ft ~ units * sq_ft + boro'
f3 = 'value_per_sq_ft ~ units + sq_ft * boro + type'
f4 = 'value_per_sq_ft ~ units + sq_ft * boro + sq_ft * type'
f5 = 'value_per_sq_ft ~ boro + type'
glm1 = smf.glm(f1, data=housing).fit()
glm2 = smf.glm(f2, data=housing).fit()
glm3 = smf.glm(f3, data=housing).fit()
glm4 = smf.glm(f4, data=housing).fit()
glm5 = smf.glm(f5, data=housing).fit()
glm_anova = anova_deviance_table(glm1, glm2, glm3, glm4, glm5)
print(glm_anova)

# do the same thing with logistic regression

housing['high_value'] = (housing['value_per_sq_ft'] >= 150).astype(int)

# create and fit our logistic regression using GLM

f1 = 'high_value ~ units + sq_ft + boro'
f2 = 'high_value ~ units * sq_ft + boro'
f3 = 'high_value ~ units + sq_ft * boro + type'
f4 = 'high_value ~ units + sq_ft * boro + sq_ft * type'
f5 = 'high_value ~ boro + type'
logistic = statsmodels.genmod.families.family.Binomial(link=statsmodels.genmod.families.links.logit)
glm1 = smf.glm(f1, data=housing, family=logistic).fit()
glm2 = smf.glm(f2, data=housing, family=logistic).fit()
glm3 = smf.glm(f3, data=housing, family=logistic).fit()
glm4 = smf.glm(f4, data=housing, family=logistic).fit()
glm5 = smf.glm(f5, data=housing, family=logistic).fit()

# show the devinace from our GLM (logistic) models
print(anova_deviance_table(glm1, glm2, glm3, glm4, glm5))

# 14.4 K-fold cross validation

##### Import #####

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression





#####

print(housing.columns)

# get the training and test data

X_train, X_test, y_train, y_test = train_test_split(
    pd.get_dummies(housing[['units', 'sq_ft', 'boro']],
                   drop_first=True),
    housing['value_per_sq_ft'],
    test_size = 0.20,
    random_state = 42
)

# We can get a score that indicates how well our model is performing using our test data.

lr = LinearRegression().fit(X_train, y_train)
print(lr.score(X_test, y_test))

# get a fresh new housing data set:

housing = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/housing_renamed.csv')

kf = KFold(n_splits=5)
y, X = dmatrices('value_per_sq_ft ~ units + sq_ft + boro', housing)

# Next we can train and test our model on each fold.

coefs = []
scores = []
for train, test in kf.split(X):
    X_train, X_test, = X[train], X[test]
    y_train, y_test = y[train], y[test]
    lr = LinearRegression().fit(X_train, y_train)
    coefs.append(pd.DataFrame(lr.coef_))
    scores.append(lr.score(X_test, y_test))

# view results

coefs_df = pd.concat(coefs)
coefs_df.columns = X.design_info.column_names
coefs_df

# we can take a look at the average coefficient across all folds using apply and the np.mean function
print(coefs_df.apply(np.mean))

import numpy as np
print(coefs_df.apply(np.mean)) # returns an error!

print(scores)

# use cross_val_scores to calculate CV scores

model = LinearRegression()
scores = cross_val_score(model, X, y, cv = 5)

print(scores)

# If we were to compare multiple models with each other, we would compare the average
# of the scores.

print(scores.mean())

# Now we'll refit all our models using k-fold cross-validation


y1, X1 = dmatrices('value_per_sq_ft ~ units + sq_ft + boro', housing)
y2, X2 = dmatrices('value_per_sq_ft ~ units * sq_ft + boro', housing)
y3, X3 = dmatrices('value_per_sq_ft ~ units + sq_ft* boro + type', housing)
y4, X4 = dmatrices('value_per_sq_ft ~ units + sq_ft*boro + sq_ft*type', housing)
y5, X5 = dmatrices('value_per_sq_ft ~ boro + type', housing)

# fit our models

model = LinearRegression()

scores1 = cross_val_score(model, X1, y1, cv = 5)
scores2 = cross_val_score(model, X2, y2, cv = 5)
scores3 = cross_val_score(model, X3, y3, cv = 5)
scores4 = cross_val_score(model, X4, y4, cv = 5)
scores5 = cross_val_score(model, X5, y5, cv = 5)

scores_df = pd.DataFrame([scores1, scores2,scores3, scores4, scores5])
print(np.mean(scores_df, axis=1)) # model 4 has the best performance!

'''When we are working with models, it’s important to measure their performance.
Using ANOVA for linear models, looking at deviance for GLM models, and using cross-validation
are all ways we can measure error and performance when trying to pick the best model.'''

