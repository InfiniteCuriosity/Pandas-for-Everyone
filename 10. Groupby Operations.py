# 10.2 Aggregate

# 10.2.1 Basic One-Variable Grouped Aggregation

# load the gapminder dataset

import pandas as pd
df = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/gapminder.tsv', sep='\t')
df.head() # that's correct
# calculate the average life expectancy for each year

avg_life_exp_by_year = df.groupby('year').lifeExp.mean()
avg_life_exp_by_year # that worked!

# Create similar results with average population by year;
avg_pop_by_year = df.groupby('year').pop.mean()
avg_pop_by_year

# create similar result with avergae GDP by year:
avg_gdp_by_year = df.groupby('year').gdpPercap.mean()
avg_gdp_by_year

# this was done using dot notation, == using [] notation, thus:

avg_life_exp_by_year = df.groupby('year')['lifeExp'].mean()
avg_life_exp_by_year # that also works!

# get a list of unique years in the data set:

years = df.year.unique()
years # very interesting, that worked!!

continents = df.continent.unique()
continents

# Subset the data for the year 1952
y1952 = df.loc[df.year == 1952]
y1952

y1952 = df.loc[df.year == 1952,] # same results with or without the colon at the end
print(y1952)

# calculate the mean of life expectancy in 1952
mean_life_exp_1952 = y1952.lifeExp.mean()
mean_life_exp_1952

mean_pop_1952 = y1952.pop.mean() # not sure why this one does not work, it returns an error
# I think Pandas might not have a mean function, that might be numpy

y1952_mean = y1952.lifeExp.mean()
y1952_mean # there are MANY other built-in methods (key word) that Pandas can use!!

#10.2.2 Built-in Aggregation Methods

# example, we can calculate multiple summary statistics simultaneously with *describe*

# group by continent and describe each group
continent_describe = df.groupby('continent').lifeExp.describe()
continent_describe

# 10.2.3 Aggregation Functions
# can call agg or aggregate method

# in this example we use the mean function from the numpy library

import numpy as np

# calculate the average life expectancy by continent, but use the np.mean function

df.groupby('continent').lifeExp.aggregate(np.mean)
df.groupby('continent').gdpPercap.aggregate(np.mean)
df.groupby('continent').pop.aggregate(np.mean)


cont_le_agg = df.groupby('continent').lifeExp.agg(np.mean)
cont_le_agg # that worked!

# note that agg and aggregate do the same thing

cont_le_agg2 = df.groupby('continent').lifeExp.aggregate(np.mean)
cont_le_agg2 # identical results

# 10.2.3.2 Custon User Functions

# create our own function (example is a mean function)

def my_mean(values):
    n = len(values)
    # start the sum at 0
    sum = 0
    for value in values:
        sum += value
    # return the summed values divided by the number of values
    return(sum/n)
values = [*range(-100,1000,100)]
print(my_mean(values))

my_mean(values) # that works for my made up set of values!!

# we can pass our current function straight into the agg or aggregate method with np.mean

agg_my_mean = df.groupby('year').lifeExp.agg(my_mean)
agg_my_mean

agg_my_mean = df.groupby('year').lifeExp.agg(my_mean) # no parentheses after the function - strange - why not?
agg_my_mean # interesting, that worked!

'''In the following example, we will calculate the global average for average life expectancy,
diff_value, and subtract it from each grouped value.'''

def my_mean_diff(values, diff_value):
    n = len(values)
    sum = 0
    for value in values:
        sum += value
    mean = sum/n
    return(mean - diff_value)

my_mean_diff(values, -1000) # my little test works!

global_mean = df.lifeExp.mean()
print(global_mean) #that's a match!

# custom aggregation function with multiple parameters

agg_mean_diff = df.groupby('year').lifeExp.agg(my_mean_diff, diff_value = global_mean)
agg_mean_diff

# 10.2.4 Multiple functions simultaneously

# calculate the count, mean, std of the lifeExp by continent

gdf = df.groupby('year').lifeExp.agg([np.count_nonzero, np.mean, np.std])
print(gdf)

# 10.2.5 using a dict in agg/aggregate
# can pass a python dictionary or DataFrame

# 10.2.5.1 - DataFrame
'''When specifying a dict on a grouped DataFrame, the keys are the columns of the DataFrame,
and the values are the functions used in the aggregated calculation.'''

'''Use a dictionary on a dataframe to agg different columns
for each year, calculate the:
average lifeExp, median pop, and median gdpPercap'''

gdf_dict = df.groupby('year').agg({
    'lifeExp': 'mean',
    'pop': 'median',
    'gdpPercap': 'median'
})
print(gdf_dict) # that worked!

# 10.2.5.2 On a Series!

'''To have user-defined column names in the output of a grouped series calculation,
you need to rename those columns after the fact.'''

gdf = df.groupby('year')['lifeExp'].agg([np.count_nonzero,
                                         np.mean,
                                         np.std]).rename(columns = {'count_nonzero': 'count',
                                                                    'mean': 'avg',
                                                                    'std': 'std_dev'}).reset_index()
print(gdf)

# 10.3 Transform (yay!, something I know very well from my experience with R!)

# let's write a Python function to calculate a z-score:



def my_zscore(x):
    '''calculates the z-score of provided data,
    'x' is a vector or series of values'''
    return((x - x.mean()) / x.std())

transform_z = df.groupby('year').lifeExp.transform(my_zscore)
transform_z

# determine the shape of the original data frame, and our transformed data set
print(df.shape)
print(transform_z.shape)

# the SciPy library has its own zscore function. Let's use the zscore function in a groupby transform:

# import the ascore function from scipy.stats
from scipy.stats import zscore

# calculate a grouped zscore
sp_z_grouped = df.groupby('year').lifeExp.transform(zscore)
sp_z_grouped

# calculate a non-grouped z-score
sp_z_nongroup = zscore(df.lifeExp)
sp_z_nongroup

'''Our grouped results are similar. However, when we calculate the z-score outside the groupby,
we get the z-score calculated on the entire data set, not broken out by group.'''

# 10.31.1 Missing Value Example

import seaborn as sns
import numpy as np

# set the seed (I think this is the first time this has been done here)
np.random.seed(42)

# sample ten rows from tips
tips_10 = sns.load_dataset('tips').sample(10)
tips_10

# count non-missing values, broken down by sex

count_sex = tips_10.groupby('sex').count()
count_sex

'''
We have three missing values for Male, and one missing value for Female.
Now let’s calculate a grouped average, and use the grouped average to fill in the missing values.'''

def fill_na_mean(x):
    '''returns the average of a given vector'''
    avg = x.mean()
    return(x.fillna(avg))

# calcualte a mean 'total_bill' by 'sex'
total_bill_group_mean = tips_10.groupby('sex').total_bill.transform(fill_na_mean)
total_bill_group_mean

# assign that result to a new column in the original data
tips_10['fill_total_bill'] = total_bill_group_mean
print(tips_10)

# 10.4 Filter
tips = sns.load_dataset('tips')
tips

print(tips.shape)

# look at the frequency counts for the table size
print(tips['size'].value_counts())

'''The output shows that table sizes of 1, 5, and 6 are infrequent. Depending on your needs,
you may want to filter those data points out. In this example,
we want each group to consist of 30 or more observations.'''

# filter the data such that each group has more than 30 observations:
tips_filtered = tips.groupby('size').filter(lambda x: x['size'].count() >= 30)
tips_filtered

print(tips_filtered['size'].value_counts())