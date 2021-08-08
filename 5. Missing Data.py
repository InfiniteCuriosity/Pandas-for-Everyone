# just import the numpy missing values
from numpy import NaN, NAN, nan

# test for equality:

print(NaN == "True") # False
print(NAN == "False") # False
print(NaN == 0) # False
print(NaN == '') # False

# missing values are not equal to other values:

print(NaN == NaN) # False
print((NaN == nan)) # False
print(nan == NAN) # False

# Pandas has built-in methods for testing for missing values:

import pandas as pd
print(pd.isnull(NaN))
print(pd.isnull(nan))
print(pd.isnull(NAN))

# pandas has a test for non-missing values

print(pd.notnull(NaN))
pd.notnull(42)
pd.notnull('missing')

# 5.3 Where do missing values come from?

# 5.3.1 Load the data

# Load the data with the default (and missing) values:

visited_file = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_visited.csv')
print(visited_file) # that works!
visited_file.dtypes


# load the data without the default missing values:

pd.read_csv()

print(pd.read_csv(visited_file), keep_default_na='True') # I can't get that to work
print(pd.read_csv(visited_file, na_values=[''], keep_default_na=False)) # I can't get that to work

# 5.3.2 Merged Data

visited = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_visited.csv')
survey = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_survey.csv')
print(survey)
print(visited)

visited1 = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_visited.csv', keep_default_na=True)
print(visited1)

visited2 = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_visited.csv', keep_default_na=False)
print(visited2)


vs = visited.merge(survey, left_on='ident', right_on='taken')
print(vs) # that works!

vs1 = survey.merge(visited, left_on='taken', right_on='ident')
vs1


# 5.3.3, missing values in a series

num_legs = pd.Series({'goat' : 4, 'amoeba' : nan})
print(num_legs)

# Missing values in a data frame:

scientists = pd.DataFrame({
    'Name': ['Rosalind Franklin', 'William Gossett'],
    'Occupation': ['Chemist', 'Statistician'],
    'Born': ['1920-07-05', '1876-06-13'],
    'Died': ['1958-04-16', '1937-10-16'],
    'missing': [NaN, nan]})

print(scientists)

# 5.3.4 re-indexing

# example, look at only years from 2000 to 2010 in the Gapminder data:

gapminder = pd.read_csv('/Users/russellconte/Desktop/gapminder.csv')
print(gapminder)

life_exp = gapminder.groupby(['year'])['lifeExp'].mean()
print(life_exp)

# re-index the 'loc', from the code above:

print(life_exp[range(2000, 2010),]) # does not work
print(life_exp.loc[range(2000, 2010), ]) # does not work

print(life_exp.loc[range(2000, 2010), ]) # no longer supported

y2000 = life_exp[life_exp.index > 2000]
y2000

# 5.4 working with missing data

ebola = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/country_timeseries.csv')
print(ebola.count())

num_rows = ebola.shape[0]
print(num_rows)
num_missing = num_rows - ebola.count()
print(num_missing)

import numpy as np
print(np.count_nonzero(ebola.isnull()))
print(np.count_nonzero(ebola['Cases_Guinea'].isnull()))

# print a frequency table of values:

print(ebola.Cases_Guinea.value_counts(dropna=False).head())

# 5.4.2 Cleaning Missing Data

# 5.4.2.1 Recode/Replace
'''We can use the fillna method to recode the missing values to another value.
For example, suppose we wanted the missing values to be recoded as a 0.'''
print(ebola.fillna(0).iloc[0:10, 0:5])

# 5.4.2.2 Fill Forward
'''When we fill data forward, the last known value is used for the next missing value.
In this way, missing values are replaced with the last known/recorded value.'''

print(ebola.fillna(method='ffill').iloc[0:10, 0:5])

# 5.4.2.3 Fill Backward
'''When we fill data backward, the newest value is used to replace the missing data.
In this way, missing values are replaced with the newest value.'''

print(ebola.fillna(method='bfill').iloc[:,0:5].tail())

# 5.4.2.4 Interpolate
'''the interpolation in Pandas fills in missing values linearly.
Specifically, it treats the missing values as if they should be equally spaced apart.'''

print(ebola.interpolate().iloc[0:10, 0:5])

# 5.4.2.5 Drop Missing Values

print(ebola.shape)
ebola.shape

# !!if we keep only complete cases in our Ebola data set, we are left with just one row of data!!
ebola_dropna = ebola.dropna()
print(ebola_dropna)

# 5.4.3 Calculations with missing data

ebola['Cases_multiple'] = ebola['Cases_Guinea'] + ebola['Cases_Liberia'] + ebola['Cases_SierraLeone']
ebola_subset = ebola.loc[:, ['Cases_Guinea', 'Cases_Liberia', 'Cases_SierraLeone', 'Cases_multiple']]
print(ebola_subset)

print(ebola.Cases_Guinea.sum(skipna=True))