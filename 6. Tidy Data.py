# 6. Tidy Data

# 6.2.1, Keep One Column Fixed

import pandas as pd
pd.set_option('display.max_columns', None)
pew = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/pew.csv')
print(pew.iloc[:, 0:6])

'''for data analytics, the table needs to be reshaped so that we have religion, income, and count variables.'''
# we do not need to specify a value_vars since we want to pivot # all the columns except for the 'religion' column

pew_long = pd.melt(pew, id_vars='religion')# id_vars represents the variables that will remain as is.
pew_long = pd.melt(pew, id_vars='<$10k')
print(pew_long.head())
print(pew_long.tail())

pew_long = pd.melt(pew, '$40-50k')
print(pew_long.head())
print(pew_long.tail())



#We can change the defaults so that the melted/unpivoted columns are named.
pew_long = pd.melt(pew,
                   id_vars='religion',
                   var_name='income',
                   value_name='count')
print(pew_long)

# 6.2.2 Keeping multiple columns fixed

'''Not every data set will have one column to hold still while you unpivot the rest of the columns.
As an example, consider the Billboard data set.'''

billboard = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/billboard.csv')
print(billboard.iloc[0:5, 0:16])
print(billboard.iloc[307:316, 1:10])
print(billboard.iloc[-10:-1, 1:10]) # That works!! Yay!!

billboard_long = pd.melt(
    billboard,
    id_vars = ['year', 'artist', 'track', 'time', 'date.entered'],
    var_name='week',
    value_name='rating')

print(billboard_long.tail())

'''if you wanted to create a faceted plot of the weekly ratings, the facet variable would need to be a column in the dataframe.'''



# Columns with Multiple Variables

ebola = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/country_timeseries.csv')
print(ebola.columns)

# print selected rows and columns
print(ebola.iloc[:5, [0, 1, 2, 3, 10, 11]])

ebola_long = pd.melt(ebola, id_vars=['Date', 'Day'])
print(ebola_long)

# 6.3.1 Split and Add Columns Individually (simple method)

# get the variable column
# access the string methods
# split the column based on a delimiter

print(ebola)
variable_split = ebola_long.variable.str.split('_')
print(variable_split)

status_values = variable_split.str.get(0)
country_values = variable_split.str.get(1)

print(status_values[:5]) # prints the first five values

print(status_values[-5:]) # prints the last five status values

print(country_values[:5]) # prints the first five values
print(country_values)


print(country_values[-5:]) # prints the last five rows

# now add the vectors to our data frame
ebola_long['status'] = status_values
ebola_long['country'] = country_values
print(ebola_long.head())
print(ebola_long.tail())
# 6.3.2 split and combine in a single step (simple method)

variable_split = ebola_long.variable.str.split('_', expand=True)
variable_split.columns = ['status', 'country']
ebola_parsed = pd.concat([ebola_long, variable_split], axis=1)
print(ebola_parsed.head())

# 6.3.3 Split and Combine in a Single Step (combined method)

ebola_long['statuses'], ebola_long['country'] = zip(*ebola_long.variable.str.split('_'))
print(ebola_long.head())

# 6.4 Variables in both Rows and Columns

weather = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/weather.csv')
print(weather.head())
weather_melt = pd.melt(weather,
                       id_vars=['id', 'year', 'month', 'element'],
                       var_name='day',
                       value_name='temp')
print(weather_melt.head())
print(weather_melt.tail())

weather_tidy_flat = weather_melt.pivot_table(
    index = ['id', 'year', 'month', 'day'],
    columns='element',
    values='temp')
print(weather_tidy_flat)

# 6.5 Multiple Observational Unites In a Table (normalization)

print(billboard_long.head())

print(billboard_long[billboard_long.track == 'Loser'].head())

'''What we should do in this case is to place the year, artist, track, time, and date.entered in a new dataframe,
with each unique set of values being assigned a unique ID. We can then use this unique ID in a second dataframe
that represents a song, date, week number, and ranking. This entire process can be thought of as reversing the steps
in concatenating and merging data described in Chapter 4.'''


billboard_songs = billboard_long[['year', 'artist', 'track', 'time']]
print(billboard_songs)

# We can then assign a unique value to each row of data.
billboard_songs['id'] = range(len(billboard_songs))
print(billboard_songs.head(n = 10))

# Now that we have a separate dataframe about songs, we can use the newly created id
# column to match a song to its weekly ranking.

billboard_ratings = billboard_long.merge(
    billboard_songs, on=['year', 'artist', 'track', 'time'])
print(billboard_ratings)
print(billboard_ratings.shape)

# 6.6 Observational Units Across Multiple Tables

import os
import urllib.request
import requests

# code to download the data
# download only the first five data sets from the list of files

with open('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/raw_data_urls.txt', 'r') as data_urls:
    for line, url in enumerate(data_urls):
        if line == 5:
            break
        fn = url.split('/')[-1].strip()
        fp = os.path.join('...', 'data', fn)
        print(url)
        print(fp)
        urllib.request.urlretrieve(url, fp)


with open('https://s3.amazonaws.com/nyc-tlc/trip+data/fhv_tripdata_2015-01.csv', 'r') as data_urls:
    for line, url in enumerate(data_urls):
        if line == 5:
            break
    fn = url.split('/')[-1].strip()
    fp = os.path.join('..', 'data', fn)
    print(url)
    print(fp)
    urllib.request.urlretrieve(url, fp)



import glob
# get a list of the CSV files from the nyc-taxi data folder
nyc_taxi_data = glob.glob('../data/fhv_*')
print(nyc_taxi_data)
