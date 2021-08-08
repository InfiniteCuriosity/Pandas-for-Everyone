#### Pandas Data Structures ####

import pandas as pd
s = pd.Series(['banana', 42])
print(s)

# manually assign index values to row names to a series by passing in a Python list

# 2.2.1 Creating a series
s1 = pd.Series(['Wes McKinney', 'Creator of Pandas'],index = ['Person', 'Who'])
print(s1)

# 2.2.2 Create a data frame

scientists = pd.DataFrame({
    'Name': ['Rosalind Frankin', 'William Gosset'],
    'Occupation': ['Chemist', 'Statistician'],
    'Born': ['1920-07-25', '1876-06-13'],
    'Died': ['1958-04-16', '1937-10-16'],
    'Ages': [37, 61]
})
print(scientists)

# 2.3 The Series

first_row = scientists[scientists.values == 'William Gosset']
print(first_row)
age37 = scientists[scientists.values == 37]
print(age37)
first_row.index
first_row.values

# 2.3.1 The Series is ndarray-like
# get the 'Age' column

scientists
ages = scientists['Ages']
print(ages)

print(ages.mean())
print(ages.max())
print(ages.std())

# 2.3.2 Boolean Subsetting: Series

scientists1 = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/scientists.csv')
scientists1

Ages = scientists1['Age']
print(Ages)

print(Ages.mean())

print(Ages[Ages>Ages.mean()])

type(Ages[Ages>Ages.mean()])

print(Ages)

rev_ages = Ages.sort_index(ascending = False) # prints the result in descending order
print(rev_ages)

# 2.4 The Data Frame

# Boolean vectors will subset rows:

scientists1[scientists1['Age'] >60]

first_half = scientists1[:4]
last_half = scientists1[4:]
print(first_half)
print(last_half)

# 2.5 Making changes to a searies and to a data frame

# 2.5.1 adding additional columns

print(scientists1['Born'].dtype)
print(scientists1['Died'].dtype)

born_datetime = pd.to_datetime(scientists1['Born'], format = '%Y-%m-%d')
print(born_datetime)

scientists1
died_datetime = pd.to_datetime(scientists1['Died'], format = '%Y-%m-%d')
print(died_datetime)

# Add new columns with the born and died datetime values

pd.set_option('display.max_columns', None) # allows to show all columns!
scientists1['born_dt'], scientists1['died_dt'] = (born_datetime, died_datetime)
print(scientists1)

scientists1['age_days_dt'] = (scientists1['died_dt'] - scientists1['born_dt'])
print(scientists1)

#2.5.3 dropping value

scientists2 = scientists1.drop(['Age'], axis = 1)
print(scientists2)

# 2.6 Exporting and importing data

# 2.6.1 pickle

names = scientists1['Name']
print(names)

names.to_pickle('/Users/russellconte/Documents/scientist_name_series.pickle')

scientist_names_from_pickle = pd.read_pickle('/Users/russellconte/Documents/scientist_name_series.pickle')
print(scientist_names_from_pickle) # Yay it worked!! Double Yay!!

# 2.6.2 CSV files - my very old and familar friend!!

names.to_csv('/Users/russellconte/Documents/scientist_name_series.csv')
print(names.to_csv)