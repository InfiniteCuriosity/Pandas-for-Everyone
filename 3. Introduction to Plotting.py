#### 3. Introduction to Plotting (in Python) ####

import seaborn as sns
import matplotlib.pyplot as plt
anscombe = sns.load_dataset("anscombe")
print(anscombe)

dataset_1 = anscombe[anscombe['dataset'] == 'I']

plt.plot(dataset_1['x'], dataset_1['y'])
plt.show() # always remember to use this line of code to see the plot - AGH!!!
plt.plot(dataset_1['x'], dataset_1['y'], 'o')
plt.show()
dataset_2 = anscombe[anscombe['dataset'] == 'II']
dataset_3 = anscombe[anscombe['dataset'] == 'III']
dataset_4 = anscombe[anscombe['dataset'] == 'IV']

fig= plt.figure() # nothing shows up!

# subplot has two rows and two columns, plot location 1:
axes1 = fig.add_subplot(2,2,1)

# subplot has two rows and two columns, plot location 2:
axes2 = fig.add_subplot(2,2,2)

# subplot has two rows and two columns, plot location 3:
axes3 = fig.add_subplot(2,2,3)

# subplot has two rows and two columns, plot location 4:
axes4 = fig.add_subplot(2,2,4)

axes1.plot(dataset_1['x'], dataset_1['y'], 'o')
axes2.plot(dataset_2['x'], dataset_2['y'], 'o')
axes3.plot(dataset_3['x'], dataset_3['y'], 'o')
axes4.plot(dataset_4['x'], dataset_4['y'], 'o')

# add a small title to each subplot:

axes1.set_title("dataset_1")
axes2.set_title("dataset_2")
axes3.set_title("dataset_3")
axes4.set_title("dataset_4")

fig.suptitle("Anscombe Data")

fig.tight_layout()

plt.show()

# 3.3 Statistical Graphics Using matplotlib


tips = sns.load_dataset("tips")
tips.head() # total of 244 rows, 7 columns

# Histogram

fig = plt.figure()
axes1 = fig.add_subplot(1,1,1)
axes1.hist(tips['total_bill'], bins = 50)
axes1.set_title('Histogram of Total Bill')
axes1.set_xlabel('Frequency')
axes1.set_ylabel('Total Bill')
fig.show()

# 3.3.2 Bivariate

# 3.3.2.1 Scatterplot

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(tips['total_bill'], tips['tip'])
axes1.set_title('Scatterplot of Total Bill vs Tip')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')
scatter_plot.show()

# 3.3.2.2 Boxplot
boxplot = plt.figure()
axes1 = boxplot.add_subplot(1,1,1)
axes1.boxplot(
    [tips[tips['sex'] == 'Female']['tip'], tips[tips['sex'] == 'Male']['tip']],
    labels = ['Female', 'Male']) # returns an error if not run all at once
axes1.set_xlabel('Sex')
axes1.set_ylabel('Tip')
axes1.set_title('Boxplot of tips by Sex')
boxplot.show()

# 3.3.3 plotting multivariate data

# use color to add more detail to the plot

def recode_sex(sex):
    if sex == 'Female':
        return 0
    else:
        return 1

tips['sex_color'] = tips['sex'].apply(recode_sex)

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(
    x = tips['total_bill'],
    y = tips['tip'],
    s = tips['size'] * 10,
    c = tips['sex_color'],
    alpha = 0.5)

axes1.set_title('Total Bill vs Tip Colored by Sex and Sized by Size')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')
scatter_plot.show()

#### 3.4 Seaborn (plotting tool in Python)

# 3.4.1 Univariate

hist, ax = plt.subplots()

ax = sns.distplot(tips['total_bill'])
ax.set_title('Total Bill Histogram with Density Plot')
plt.show()

# if we only want a historgram, set kde to 'False'

hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], kde= False, bins = 10)
ax.set_title('Total Bill Histogram')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Frequency')
plt.show()

# 3.4.1.2 Density Plot (Kernel Density Estimation)

den, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], hist=False)
ax.set_title('Total Bill Histogram')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Frequency')
plt.show()

# 3.4.3.1 Rug Plots

#hist_den_rug, ax = plt.subplots -> error because I forgot the () !!!
hist_den_rug, ax = plt.subplots()
ax = sns.countplot('day', data = tips)
ax.set_title('Count of days')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Frequency')
plt.show()

# 3.4.1.4 County Plot (Bar Plot)

count, ax = plt.subplots()
ax = sns.countplot('day', data = tips)
ax.set_title('Count of days')
ax.set_xlabel('Day of the week')
ax.set_ylabel('Frequency')
plt.show()
