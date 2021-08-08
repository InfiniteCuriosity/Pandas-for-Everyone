import pandas as pd
import seaborn as sns

tips = sns.load_dataset("tips")
print(tips.dtypes)

# 7.3.1 Converting to String Objects
'''To convert values into strings, we use the astype1 method on the column. astype takes a parameter, dtype,
that will be the new data type the column will take on.
In this case, we want to convert the sex variable to a string object, str.'''

tips['sex_str'] = tips['sex'].astype(str)

print(tips.dtypes)
print(tips.head())
print(tips.values)

# 7.3.2 Converting to numeric values

# convert total_bill into a string

tips['total_bill'] = tips['total_bill'].astype(str)
print(tips.head())
print(tips.dtypes)

# convert it back to a float
tips['total_bill'] = tips['total_bill'].astype(float)
print(tips.dtypes)

# 7.3.2 1 to numeric

'''Let’s subset our tips dataframe and also put in a 'missing' value in the total_bill column
to illustrate how the to_numeric function works.'''

# subset the tips data

tips_sub_miss = tips.head(10)
print(tips_sub_miss)
# assign some missing values:
tips_sub_miss.loc[[1,3,5,7], 'total_bill'] = 'missing'

print(tips_sub_miss) # that worked!

# Looking at the dtypes, you will see that the total_bill column is now a string object.

print(tips_sub_miss.dtypes) # that's correct!

''' if we pass errors the 'ignore' value, nothing will change in our column.
But we also do not get an error message—which may not always the behavior we want.'''

tips_sub_miss['total_bill'] = pd.to_numeric(tips_sub_miss['total_bill'],errors = 'ignore')
print(tips_sub_miss)

'''In contrast, if we pass in the 'coerce' value, we will get NaN values for the 'missing' string.'''

tips_sub_miss['total_bill'] = pd.to_numeric(tips_sub_miss['total_bill'], errors='coerce')
print(tips_sub_miss) # that works!

# 7.3.2.2 to_numeric downcast

'''The to_numeric function has another parameter called downcast, which allows you to change (i.e., “downcast”) the numeric dtype to
the smallest possible numeric dtype after a column (or vector) has been successfully converted to a numeric vector.'''

tips_sub_miss['total_bill'] = pd.to_numeric(
    tips_sub_miss['total_bill'],
    errors='coerce',
    downcast='float')

print(tips_sub_miss)
print(tips_sub_miss.dtypes) # interesting that total_bill is now float32!

# 7.4 Categorical Data

# To convert a column into a categorical type, we pass category into the astype method.

# convert the sex column into a string object first
tips['sex'] = tips['sex'].astype('str')
# now convert it back into categorical data
tips['sex'] = tips['sex'].astype('category')
print(tips.info)
print(tips.dtypes)
print(tips.info())

# 7.5 Conclusion

'''This chapter covered how to convert from one data type to another. dtypes govern which operations
can and cannot be performed on a column. While this chapter is relatively short,
converting types is an important skill when you are working with data and when you are using other Pandas methods.'''