import pandas as pd

df = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/gapminder.tsv', sep = '\t')
df.head() # shows the first five rows of the dataframe

df.shape # returns the tuple: (1704, 6), which means 1704 rows, 6 columns

df.columns # tells us names of the columns are in this data frame

df.dtypes # tells us the *types* of data in each column

df.info #-> returns the data frame
df.info() # -> returns information about the dataframe

#### Subsetting columns ####

# by name
country_df = df['country']
country_df # lists country names

country_df.head() # provides the first five rows of the data frame
country_df.tail() # provides the last five rows of the data frame

# Subsetting multiple columns, and returning them in a specific sequence/order

subset = df[['country', 'continent', 'year']]
subset

subset1 = df[['year', 'continent', 'country']]
subset1

#### Subsetting rows ####

# print the first row:

df.loc[0]

# print the 100th row:

df.loc[99]

# print the last row:
df.tail(n = 1)

subset_loc = df.loc[0]
subset_loc

# this prints differently than the previous method:
subset_head = df.head(n = 1)
subset_head

type(subset_loc) # returns the type of subset_loc

type(subset_head)

### Subsetting multiple rows ###
# select the first, 100th, and 1000th row
smaller = df.loc[[0, 99, 999]]
smaller # view the result

# we can use iloc to find the last row (something we/I could not do with loc):

df.iloc[[-1]]

##### 1.3.3 Mixing it up ####

# 1.3.3.1 Subsetting columns

subset2 = df.loc[:,['year', 'pop']] # the colon is used to select all rows
subset2

subset3 = df[['year', 'pop']]
subset3 # accomplishes the same thing

subset4 = df.iloc[:,[2,4,-1]]
subset4

# 1.3.3.2 subsetting columns by range:

small_range = list(range(5))
small_range

subset5 = df.iloc[:, small_range]
subset5

subset6 = df.iloc[:,[3,4,5]]
subset6

## 1.3.3.3 slicing columns

df.iloc[:, :3]# the first three columns:

df.iloc[:, 2:5] # columns 2, 3, and 4

df.iloc[:,0:3]

df.iloc[:,0:4:]
df.iloc[:, 0::2]
df.iloc[:,:6:2]
df.iloc[:,::2]

df.loc[42, 'country']

# 1.3.3.5 subsetting multiple rows and columns

df.iloc[[0,99,999], [ 5,3,0]] # by numbers

df.loc[[0,99,999], ['gdpPercap', 'lifeExp', 'country']] # by column names

### 1.4 Grouped and Aggregated Calculations

# 1.4.1 Grouped means

# for each year in our data, what is the average life expectancy?

df.groupby('year')['lifeExp'].mean() # this works!!

multi_group_var = df.groupby(['year', 'continent'])[['lifeExp', 'gdpPercap']].mean()
multi_group_var

# 1.5 Basic plot

global_yearly_life_expectancy = df.groupby('year')['lifeExp'].mean()
global_yearly_life_expectancy

