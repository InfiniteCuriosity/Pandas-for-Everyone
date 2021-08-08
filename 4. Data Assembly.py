# 4.3 Concatentation

# 4.3.1 Adding Rows

import pandas as pd
df1 = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/concat_1.csv')
df2 = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/concat_2.csv')
df3 = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/concat_3.csv')
print(df1)
print(df2)
print(df3)

# All of the dataframes to be concatenated are passed in a list.

row_concat = pd.concat([df1, df2, df3])
print(row_concat) # that worked!

# practice subsetting the data frame, as per chapter 2:

print(row_concat.iloc[3,]) # returns the 4th row, because Python is 0-indexed

# attempt to add a new row to the data frame:

new_row_series = pd.Series(['n1', 'n2', 'n3', 'n4'])
print(new_row_series)

# attempt to add a new row to the data frame:

print(pd.concat([df1, new_row_series])) # did not add a row, it added items in a new column

# to fix the problem, first turn our data into a data frame:

new_row_df = pd.DataFrame([['n1', 'n2', 'n3', 'n4']], columns= ['A', 'B', 'C', 'D'])
print(new_row_df)

print(pd.concat([df1, new_row_df])) # this worked!

# it's possible to append a single object to a data frame. For example:

print(df1.append(df2)) # that worked!

# using a single-row data frame:

print(df1.append(new_row_df))

# Using a Python dictionary:

data_dict = {'A': 'n1',
             'B': 'n2',
             'C': 'n3',
             'D': 'n4'}

print(df1.append(data_dict, ignore_index=True)) # that worked!

# 4.3.1.1 Ignoring the Index

#If we simply want to concatenate or append data together,
# we can use the ignore_index parameter to reset the row index after the concatenation.

row_concat_i = pd.concat([df1, df2, df3],ignore_index=True)
print(row_concat_i)

# 4.3.2 Adding columns

# set axis = 1 to concat columns; axis = 0 concatenates rows

col_concat = pd.concat([df1, df2, df3],axis=1)
print(col_concat)
#### Note this can generate multiple colomns with the exact same name####

print(col_concat['A']) # Wow, that's really, really bad!!

#Adding a single column to a dataframe can be done directly without using any specific
# Pandas function. Simply pass a new column name the vector you want assigned to
# the new column.

col_concat['new_col_list'] = ['n1', 'n2', 'n3', 'n4']
print(col_concat)

col_concat['new_col_series'] = pd.Series(['n1', 'n2', 'n3', 'n4'])
print(col_concat)

print(pd.concat([df1, df2, df3], axis=1, ignore_index=True))

# 4.3.3 Concatenation with Different Indices

# 4.3.3.1 Concatenate Rows With Different Columns
# Let’s modify our dataframes for the next few examples.

df1.columns = ['A', 'B', 'C', 'D']
df2.columns = ['E', 'F', 'G', 'H']
df3.columns = ['A', 'C', 'F', 'H']
print(df1)
print(df2)
print(df3)

# If we try to concatenate these dataframes as we did in Section 4.3.1,
# the dataframes now do much more than simply stack one on top of the other.
# The columns align themselves, and NaN fills in any missing areas.

row_concat = pd.concat([df1, df2, df3])
print(row_concat) # disaster!

#If we try to keep only the columns from all three dataframes,
# we will get an empty dataframe, since there are no columns in common.

print(pd.concat([df1, df2, df3], join='inner')) # another disaster

# If we use the dataframes that have columns in common,
# only the columns that all of them share will be returned.

print(pd.concat([df1, df3], ignore_index=False, join='inner'))

# 4.3.3.2 Concatenate Columns with Different Rows

# Let’s take our dataframes and modify them again so that they have different row indices.
# Here, we are building on the same dataframe modifications from Section 4.3.3.1.

df1.index = [0, 1, 2, 3]
df2.index = [4, 5, 6, 7]
df3.index = [0, 2, 5, 7]
print(df1)
print(df2)
print(df3)

# When we concatenate along axis=1, we get the same results from concatenating along axis=0.
# The new dataframes will be added in a column-wise fashion
# and matched against their respective row indices.
# Missing values indicators appear in the areas where the indices did not align.

col_concat = pd.concat([df1,df2, df3], axis=1)
print(col_concat)

# Just as we did when we concatenated in a row-wise manner, we can choose to keep the
# results only when there are matching indices by using join='inner'.

print(pd.concat([df1, df3], join='inner'))

# 4.4 Merging Multiple Data Sets

person = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_person.csv')
site = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_site.csv')
survey = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_survey.csv')
visited = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/survey_visited.csv')

print(person)
print(site)
print(survey)
print(visited)

# We can do this with the merge function in Pandas. merge is actually a DataFrame
# method.
# When we call this method, the dataframe that is called will be referred to the one on
# the 'left'. Within the merge function, the first parameter is the 'right' dataframe.
# The next parameter is how the final merged result looks.
# Table 4.1 provides more details. Next, we set the on parameter.
# This specifies which columns to match on. If the left and right columns do not have
# the same name, we can use the left_on and right_on parameters instead.

# 4.4.1 One-to-One Merge

# In the simplest type of merge, we have two dataframes where we want to join one column to another column,
# and where the columns we want to join do not contain any duplicate values.

# For this example, we will modify the visited dataframe so there are no
# duplicated site values.

visited_subset = visited.loc[[0,2,6],]
# We can perform our one-to-one merge as follows:
o2o_merge = site.merge(visited_subset, left_on='name', right_on='site')
print(visited_subset)
print(o2o_merge)

# 4.4.2 Many-to-One Merge

# If we choose to do the same merge, but this time without using the subsetted
# visited dataframe, we would perform a many-to-one merge. In this kind of merge,
# one of the dataframes has key values that repeat. The dataframes that contains
# the single observations will then be duplicated in the merge.

m2o_merge = site.merge(visited, left_on='name', right_on='site')
print(m2o_merge)

# 4.4.3 Many-to-Many Merge

# Lastly, there will be times when we want to perform a match based on multiple columns.
# As an example, suppose we have two dataframes that come from person merged with survey,
# and another dataframe that comes from visited merged with survey.

ps = person.merge(survey, left_on='ident', right_on='person')
vs = visited.merge(survey, left_on='ident', right_on='taken')

print(ps)
print()
print(vs)

#We can perform a many-to-many merge by passing the multiple columns to match on
# in a Python list.

ps_vs = ps.merge(vs, left_on=['ident', 'taken', 'quant', 'reading'],
                 right_on=['person', 'ident', 'quant', 'reading'])

# look at the first row of data:

print(ps_vs.loc[0,])
