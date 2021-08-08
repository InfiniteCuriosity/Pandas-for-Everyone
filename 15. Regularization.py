# begining example: Linear regression

##### -- import -- #####

import pandas as pd
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
pd.set_option('max_columns', None)
from sklearn.linear_model import ElasticNetCV


#########################

acs = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/acs_ny.csv')
print(acs.columns)

response, predictors = dmatrices(
    'FamilyIncome ~ NumBedrooms + NumChildren + NumPeople + NumRooms + NumUnits + NumVehicles + '\
    'NumWorkers + OwnRent + YearBuilt + ElectricBill +'\
    'FoodStamp + HeatingFuel + Insurance + Language',
    data = acs
)

X_train, X_test, y_train, y_test = train_test_split(predictors, response, random_state=0)

'''Now, let’s fit our linear model. Here we are normalizing our data so we can compare our coefficients when we use our regularization techniques.'''

lr = LinearRegression(normalize=True).fit(X_train, y_train)

model_coefs = pd.DataFrame(list(zip(predictors.design_info.column_names, lr.coef_[0])), columns=['variable', 'coef_lr'])
print(model_coefs)

# now let's look at how our model scores:

print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

# 15.3 LASSO regression

# LASSO, which stands for least absolute shrinkage and selection operator.

lasso = Lasso(normalize=True, random_state=0).fit(X_test, y_test)

# Now, let’s get a dataframe of coefficients, and combine them with our linear regression results.

coefs_lasso = pd.DataFrame(
    list(zip(predictors.design_info.column_names, lasso.coef_)),
    columns=['variable', 'coef_lasso']
)

model_coefs = pd.merge(model_coefs, coefs_lasso, on='variable')
print(model_coefs)

# Finally, let’s look at our training and test data scores.

print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))

# 15.4 Ridge Regression

'''Most of the code will be very similar to that seen with the previous methods.
We will fit the model on our training data, and combine the results with our ongoing dataframe of results.'''

ridge = Ridge(normalize=True, random_state=0).fit(X_train, y_train)

coefs_ridge = pd.DataFrame(
    list(zip(predictors.design_info.column_names, ridge.coef_[0])),
    columns=['variable', 'coef_ridge']
)

model_coefs = pd.merge(model_coefs, coefs_ridge, on='variable')
print(model_coefs)

# 15.5 Elastic Net

# The elastic net is a regularization technique that combines the ridge and LASSO regression techniques.

en = ElasticNet(random_state=42).fit(X_train, y_train)

coef_en = pd.DataFrame(
    list(zip(predictors.design_info.column_names, en.coef_)),
    columns=['variable', 'coef_en']
)

model_coefs = pd.merge(model_coefs, coef_en, on='variable')
pd.set_option('max_columns', None)
print(model_coefs)

# 15.6 Cross-Validation

en_cv = ElasticNetCV(cv = 5, random_state=42).fit(X_train, y_train)

coefs_en_cv = pd.DataFrame(
    list(zip(predictors.design_info.column_names, en_cv.coef_)),
    columns = ['variable', 'coef_en_cv']
)
model_coefs = pd.merge(model_coefs, coefs_en_cv, on='variable')
print(model_coefs)

'''Regularization is a technique used to prevent overfitting of data.
It achieves this goal by applying some penalty for each feature added to the model.
The end result either drops variables from the model or decreases the coefficients of the model.
Both techniques try to fit the training data less accurately but hope to provide better predictions with
data that has not been seen before. These techniques can be combined (as seen in the elastic net),
and can also be iterated over and improved with cross-validation.'''