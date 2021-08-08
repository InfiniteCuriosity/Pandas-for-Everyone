import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn import linear_model
import numpy as np

tips = sns.load_dataset('tips')
print(tips.head())
tips.info()

# using statsmodels
model = smf.ols(formula='tip ~ total_bill', data = tips)

#Once we have specified our model, we can fit the data to the model by using the fit method.
results = model.fit()
print(results.summary())
'''We can use these parameters in our formula for the line, y = (0.105)x + 0.920.
To interpret these numbers, we say that for every one unit increase in total_bill
(i.e., every time the bill increases by a dollar),
the tip increases by 0.105 (i.e., 10.5 cents).'''

# If we just want the coefficients, we can call the params attribute on the results.
print(results.params) # that works!

# create the LinearRegression object

lr = linear_model.LinearRegression()

'''Next, we need to specify the predictor, X, and the response, y.
To do this, we pass in the columns we want to use for the model.'''

predicted = lr.fit(X = tips['total_bill'].values.reshape(-1,1), y = tips['tip'])
print(predicted.coef_)
print(predicted.intercept_)

# 12.3 Multiple Regression

#12.3.1 Using Statsmodels

model = smf.ols(formula = 'tip ~ total_bil', data = tips).fit()
print(model.summary()) # I GREATLY prefer the results using this method over sklearn!

'''The interpretations are exactly the same as before, although each parameter is interpreted
“with all other variables held constant.” That is, for every one unit increase (dollar) in total_bill,
the tip increases by 0.09 (i.e., 9 cents) as long as the size of the group does not change.'''

# Here’s the model that uses all the variables in our data.
model = smf.ols(formula = 'tip ~ total_bill + size + sex + smoker + day + time', data = tips).fit()
print(model.summary())

'''For example, the coefficient for sex[T.Female] is 0.0324. We interpret this value in relation to
the reference value, Male; that is, we say that when the sex changes from Male to Female,
the tip increases by 0.324.'''

# 12.3.3 Using Sklearn

lr = linear_model.LinearRegression()
predicted = lr.fit(X = tips[['total_bill', 'size']], y = tips['tip'])
print(predicted.coef_)
print(predicted.intercept_)

# 12.3.4 Using sklearn With Categorical Variables

#To drop the reference variable, we can pass in drop_first=True.
tips_dummy = pd.get_dummies(
    tips[['total_bill', 'size', 'sex', 'smoker', 'day', 'time']], drop_first=True)

x_tips_dummy_ref = pd.get_dummies(
     tips[['total_bill', 'size', 'sex', 'smoker', 'day', 'time']], drop_first=True)

print(tips_dummy.head())

# we fit the model just as we did earlier

lr = linear_model.LinearRegression()
predicted = lr.fit(X = x_tips_dummy_ref, y = tips['tip'])
print(predicted.coef_)
print(predicted.intercept_)

# 12.4 Keeping Index Labels from sklearn

# create and fit the linear model
lr = linear_model.LinearRegression()
predicted = lr.fit(X = x_tips_dummy_ref, y = tips['tip'])

# get the intercept along with other coefficients
values = np.append(predicted.intercept_, predicted.coef_)

# get the names of the values
names = np.append('intercept', tips_dummy.columns)

# put everything in a labeled data frame
results = pd.DataFrame(values, index = names, columns = ['coef'])

print(results)