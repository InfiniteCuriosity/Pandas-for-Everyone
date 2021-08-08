# 12. Linear Models

# 12.2 Simple Linear Regression

import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips.head())

# 12.2.1 use statsmodels to perform our simple linear regression

import statsmodels.formula.api as smf
model = smf.ols(formula='tip ~ total_bill', data = tips)

results = model.fit()

print(results.summary()) # returns a summary of the model

print(results.conf_int())

# 12.2.2 Using sklearn

# start by importing linear models from sklearn

from sklearn import linear_model
# create our linear regression object

lr = linear_model.LinearRegression()
'''Next, we need to specify the predictor, X, and the response, y. To do this, we pass in the
columns we want to use for the model.'''

'''Calling reshape directly on the column will raise either a DeprecationWarning (Pandas 0.17) or a ValueError (Pandas 0.19),
depending on the version of Pandas being used. To properly reshape our data, we must use the values attribute
(otherwise you may get another error or warning). When we call values on a Pandas dataframe or series, we get the
numpy ndarray representation of the data.'''

predicted = lr.fit(X = tips['total_bill'].values.reshape(-1, 1), y = tips['tip'])
print(predicted.coef_)
print(predicted.intercept_)
# note people in our sample are tipping around 10% of their bill amount

# 12.3, Multiple Regression

# 12.3.1 Using statsmodels

model = smf.ols(formula = 'tip ~ total_bill + size', data = tips).fit()
print(model.summary())

# 12.3.2 Using stats models with Categorical Variables

'''To this point, we have used only continuous predictors in our model. If we look at the info attribute of our tips data set,
however, we can see that our data includes categorical variables.'''

print(tips.info())

print(tips.sex.unique())

'''Here’s the model that uses all the variables in our data.'''

model = smf.ols(formula= 'tip ~ total_bill + size +sex + smoker + day + time', data = tips).fit()
print(model.summary())

# 12.3.3, using sklearn with Categorical Variables

'''The syntax for multiple regression in sklearn is very similar to the syntax for simple linear regression with this library.
To add more features into the model, we pass in the columns we want to use.'''

lr = linear_model.LinearRegression()

tips_dummy = pd.get_dummies(tips[['total_bill', 'size', 'sex', 'smoker', 'day', 'time']])

# To drop the reference variable, we can pass in drop_first=True.
x_tips_dummy_ref = pd.get_dummies(
     tips[['total_bill', 'size',
'sex', 'smoker', 'day', 'time']], drop_first=True)


print(tips_dummy.head())

'''We fit the model just as we did earlier.'''

lr = linear_model.LinearRegression()
predicted = lr.fit(X = x_tips_dummy_ref, y = tips['tip'])
print(predicted.coef_)