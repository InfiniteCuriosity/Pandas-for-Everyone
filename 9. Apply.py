# 9.2 Functions

#def my_function():
    # indent four spaces
    # function code here

def my_sq(x):
    return(x**2)

my_sq(3) # testable example

def avg_2(x,y):
    return((x+y)/2)

avg_2(5,9) # that works!

# 9.3 Apply (Basics)

import pandas as pd
df = pd.DataFrame({'a': [10, 20, 30],
                   'b': [20, 30, 40]})

print(df)

print(df['a']**2)

# 9.3.1 Apply Over a Series

'''In our example, if we subset a single column or row, the type of the object we get back is a Pandas Series.'''

# get the first column:
print(df['a'])

# print the first row:

df.iloc[0,] # that works!

# apply our square function on the 'a' column:

sq = df['a'].apply(my_sq) # note we do NOT need rounded brackets when we pass the function to apply
print(sq)

def my_exp(x, e):
    return x**e

my_exp(5,-1)

# to apply the function on our series, we will need to pass in the second parameter

ex = df['a'].apply(my_exp, e = -1)
print(ex)

# 9.3.2 Apply Over a DataFrame

# write a function that takes a single value, and prints out the given value:

def print_me(x):
    print(x)

# column wise -> axis = 0
# row wise -> axis = 1

df.apply(print_me, axis=0)

# compare to the following output:

print(df['a'])
print(df['b'])

'''let’s write a function that calculates the mean (average) of three numbers'''

def mean_3(x,y,z):
    return((x+y+z)/3)

mean_3(1000,10000,100000)

# 9.3.2 Row-wise operations

def avg_2_apply(row):
    x = row[0]
    y = row[1]
    return (x+y)/2

df.apply(avg_2_apply, axis=0)

# 9.4 Apply (More Advanced)
import seaborn as sns

pd.set_option('display.max_columns', None)

titanic = sns.load_dataset("titanic")
print(titanic.info())

# calculate the number of missing values, and the number of NaN values

import numpy as np
def count_missing(vec):
    """Counts the number of missing values in a vector """
    # get a vector of True/False values
    # depending whether the value is missing
    null_vec = pd.isnull(vec)
    # take the sum of the null_vec
    # since null values do not contribute to the sum
    null_count = np.sum(null_vec)
    # return the number of missing values in the vector
    return null_count

def prop_missing(vec):
    """Percentage of missing values in a vector"""
    # numerator: number of missing values
    # we can use the count_missing function we just wrote!
    num = count_missing(vec)
    # denominator: total number of values in the vector
    # we also need to count the missing values
    dem = vec.size
    # return the proportion/percentage of missing
    return num / dem

def prop_complete(vec):
    """Percentage of nonmissing values in a vector"""
    # we can utilize the percent_missing function we just wrote
    # by subtracting its value from 1
    return 1 - prop_missing(vec)

# 9.4.1 Column-wise Operations

cmis_col = titanic.apply(count_missing)

pmis_col = titanic.apply(prop_missing)

pcom_col = titanic.apply(prop_complete)

print(cmis_col)

print(pmis_col)

print(pcom_col)

'''For example, there are only two missing values in the embark_town column.
We can easily check those rows to see if these values are missing randomly,
or if there is a special reason for them to be missing.'''

print(titanic.loc[pd.isnull(titanic.embark_town), :])

# 9.4.2 Row-wise operations

cmis_row = titanic.apply(count_missing, axis = 1)

pmis_row = titanic.apply(prop_missing, axis=1)

pcom_row = titanic.apply(prop_complete, axis = 1)

print(cmis_row.head())

print(pmis_row.head())

print(pcom_row.head())

# check if any rows in our data have multiple missing values:

print(cmis_row.value_counts())

# since we are using Apply in a row-wise manner, we can actually create a new column containing these values:

titanic['num_missing'] = titanic.apply(count_missing, axis = 1)

print(titanic.head)

'''We can then look at the rows with multiple missing values.
Since there are too many rows with multiple values to print in this book,
let’s randomly sample the results.'''

print(titanic.loc[titanic.num_missing>1, :].sample(10))

# 9.5 Vectorized functions

print(df)

print(avg_2(df['a'], df['b']))

# Let’s change our function and perform a non-vectorizable calculation.

import numpy as np
def avg_2_mod(x,y):
    ''' calculate the average, unless x = 20'''
    if (x == 20):
        return(np.NaN)
    else:
        return((x+y)/2)

avg_2_mod(10, 20)

# 9.5.1 Using Numpy

# vectorize to create a new function

avg_2_mod_vec = np.vectorize(avg_2_mod)
print(avg_2_mod_vec(df['a'], df['b']))

'''Decorators are “functions” that take another function as input, and modify how that function’s output behaves.'''

@np.vectorize
def v_avg_2_mod(x, y):
    if (x == 20):
        return(np.NaN)
    else:
        return (x+y) / 2

print(v_avg_2_mod(df['a'], df['b'])) # returns an error, I'm not sure why!

# 9.5.2 Using numba

import numba

def v_avg_2_numba(x,y):
    if (int(x) == 20):
        return(np.NaN)
    else:
        return (x+y) /2

print(v_avg_2_numba(df['a'], df['b']))
