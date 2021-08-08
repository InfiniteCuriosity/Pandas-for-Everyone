# Introduction to Plotting

import seaborn as sns
import matplotlib.pyplot as plt

anscombe = sns.load_dataset("anscombe")
print(anscombe)

# 3.2 Matplotlib
dataset_1 = anscombe[anscombe['dataset'] == 'I']
dataset_2 = anscombe[anscombe['dataset'] == 'II']
dataset_3 = anscombe[anscombe['dataset'] == 'III']
dataset_4 = anscombe[anscombe['dataset'] == 'IV']

plt.plot(dataset_1['x'], dataset_1['y'])  # will print with lines
plt.show()

plt.plot(dataset_1['x'], dataset_1['y'], 'o')  # will print with blue dots
plt.show()

fig = plt.figure()

axes1 = fig.add_subplot(2, 2, 1)  # 2 rows, 2 columns, plot location 1
axes2 = fig.add_subplot(2, 2, 2)  # 2 rows, 2 columns, plot location 2
axes3 = fig.add_subplot(2, 2, 3)  # 2 rows, 2 columns, plot location 3
axes4 = fig.add_subplot(2, 2, 4)  # 2 rows, 2 columns, plot location 4

# add a plot to each of the axes created above
axes1.plot(dataset_1['x'], dataset_1['y'], 'o')
axes2.plot(dataset_2['x'], dataset_2['y'], 'o')
axes3.plot(dataset_3['x'], dataset_3['y'], 'o')
axes4.plot(dataset_4['x'], dataset_4['y'], 'o')

# add a small title to each of the four subplots
axes1.set_title("dataset_1")
axes2.set_title("dataset_2")
axes3.set_title("dataset_3")
axes4.set_title("dataset_4")

# add a title for the entire figure:
fig.suptitle("Anscombe Data")

# add a tight layout:
fig.tight_layout()
plt.show()

# 3.3 Statistical Graphs Using Matplotlib

tips = sns.load_dataset("tips")
print(tips.head())
tips.info # 244 rows, 7 columns
# 3.3.1 Univariate

# 3.3.3.1 Histograms

fig = plt.figure()
axes1 = fig.add_subplot(1, 1, 1)
axes1.hist(tips['total_bill'], bins=100)
axes1.set_title('Histogram of Total Bill')
axes1.set_xlabel('Frequency')
axes1.set_ylabel('Total Bill')
fig.show()

# 3.3.2 Bivariate

# 3.3.2.1 Scatterplot

scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1, 1, 1)
axes1.scatter(tips['total_bill'], tips['tip'])
axes1.set_title('Scatterplot of Total Bill vs Tip')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')
scatter_plot.show()

# 3.3.2.2 Boxplot

boxplot = plt.figure()
axes1 = boxplot.add_subplot(1, 1, 1)
axes1.boxplot(
    [tips[tips['sex'] == 'Female']['tip'],
     tips[tips['sex'] == 'Male']['tip']],
    labels=['Female', 'Male'])
axes1.set_xlabel('Sex')
axes1.set_ylabel('Tip')
axes1.set_title('Boxplot of Tips by Sex')
boxplot.show()


# 3.3.3 Multivariate Data

# crate a color based on sex

def recode_sex(sex):
    if sex == 'Female':
        return 0
    else:
        return 1


tips['sex_color'] = tips['sex'].apply(recode_sex)
scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1, 1, 1)
axes1.scatter(
    x=tips['total_bill'],
    y=tips['tip'],

    # set the size of the dots based on party size
    # multiply the value by 10 to make the points bigger and easier to see
    s=tips['size'] * 10,

    # set the color for the sex
    c=tips['sex_color'],

    # set the alpha value so points are more transparent
    alpha=0.5)

axes1.set_title('Total bill vs Tip Colored by Sex and Sized by Size')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')
scatter_plot.show()

# 3.4 Seaborn

# 3.4.1 Univariate

# 3.4.1.1 Histograms
# Histograms are crated using sns.distplot

hist, ax = plt.subplots()

# use the distplot function from seaborn to create our plot
ax = sns.distplot(tips['total_bill'])
ax.set_title('Total Bill Histogram with Density Plot')
plt.show()  # shows both a histogram and a density plot

# Show a histogram only

hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], kde=False)
ax.set_title('Total Bill Histogram')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Unit Probability')
plt.show()

# 3.4.1.2 Density Plot (Kernel Density Estimation)

den, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], hist=False)
ax.set_title('Total Bill Density')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Unit Probability')
plt.show()

# 3.4.1.3 Rug Plot = A One-Dimensional Representation of a Variable's Distribution

hist_den_run, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], rug=True)
ax.set_title('Total Bill Histogram with Density and Rug Plot')
ax.set_xlabel('Total Bill')
plt.show()

# 3.4.1.4 Count Plot (Bar Plot)

count, ax = plt.subplots()
ax = sns.countplot('day', data=tips)
ax.set_title('Count of days')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Frequency')
plt.show()

# 3.4.2 Bivariate Data

# 3.4.2.1 Scatterplot

scatter, ax = plt.subplots()
ax = sns.regplot(x='total_bill', y='tip', data=tips)
ax.set_title('Scatterplot of Total Bill and Tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
plt.show()

# lmplot - creates a figure, whereas regplot creates axes

fig = sns.lmplot(x='total_bill', y='tip', data=tips)
plt.show()

# create a scatterplot that includes a univariate plot on each axis

join = sns.jointplot(x='total_bill', y='tip', data=tips)
join.set_axis_labels(xlabel='Total Bill', ylabel='Tip')
join.fig.suptitle('Joint Plot of Total Bill and Tip', fontsize=10, y=1.03)
plt.show()

# 3.4.2.2 Hexabin Plot

hexabin = sns.jointplot(x='total_bill', y='tip', data=tips, kind="hex")
hexabin.set_axis_labels(xlabel='Total Bill', ylabel='Tip')
hexabin.fig.suptitle('Hexbin Joint Plot of Total Bill and Tip', fontsize=10, y=1.03)
plt.show()

# 3.4.2.3 2D Density Plot

kde, ax = plt.subplots()
ax = sns.kdeplot(data=tips['total_bill'],
                 data2 = tips['tip'],
                 shade = True)
ax.set_title('Kernel Density Plot of Total Bill and Tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
plt.show()

# provides density plots on both axes
kde_joint = sns.jointplot(x = 'total_bill', y = 'tip', data = tips, kind = 'kde')
plt.show()

# 3.4.2.4 Multiple Bar Plots on one result

bar, ax = plt.subplots()
ax=sns.barplot(x = 'time', y = 'total_bill', data = tips)
ax.set_title('Bar plot of average total bill for time of day')
ax.set_xlabel('Time of day')
ax.set_ylabel('Average total bill')
plt.show()

# 3.4.2.5 Boxplot

# Unlike the previously mentioned plots, a boxplot (Figure 3.24) shows
# multiple statistics: the minimum, first quartile, median, third quartile,
# maximum, and, if applicable, outliers based on the interquartile range.

box, ax = plt.subplots()
ax = sns.boxplot(x = 'time', y = 'total_bill', data = tips)
ax.set_title('Boxplot of total bill by time of day')
ax.set_xlabel('Time of day')
ax.set_ylabel('Total Bill')
plt.show()

# 3.4.2.6 Violin Plt

#  Violin plots (Figure 3.25) are able to show the same values as a boxplot,
#  but plot the “boxes” as a kernel density estimation.

violin, ax = plt.subplots()
ax = sns.violinplot(x = 'time', y = 'total_bill', data = tips)
ax.set_title('Violin plot of total bill by time of day')
ax.set_xlabel('Time of day')
ax.set_ylabel('Total Bill')
plt.show()

# 3.4.2.7 Pairwise relationships

fig = sns.pairplot(tips)
pair_grid = sns.PairGrid(tips)
# we can use plt.scatter instead of sns.regplot
pair_grid = pair_grid.map_upper(sns.regplot)
pair_grid = pair_grid.map_lower(sns.kdeplot)
pair_grid = pair_grid.map_diag(sns.distplot, rug = True)
plt.show()

# 3.4.3 Multivariate Data

# As mentioned in Section 3.3.3, there is no de facto template for plotting
# multivariate data. Possible ways to include more information are to use
# color, size, and shape to distinguish data within the plot.

# 3.4.3.1 Colors
violin, ax = plt.subplots()
ax = sns.violinplot(x = 'time', y = 'total_bill',
                    hue = 'sex', data = tips, split = True)
plt.show()

# The hue parameter can be passed into various other plotting functions

# note the use of lmplot instead of regplot here
scatter = sns.lmplot(x ='total_bill', y = 'tip', data = tips, hue = 'sex',
                     fit_reg = False)
plt.show()

# 3.4.3.2 Size and Shape

# the code below returns an error
scatter = sns.lmplot(x = 'total_bill', y = 'tip',
                     data = tips, fit_reg = False,
                     hue = 'sex',
                     scatter_kws={'s':tips['size']*10})
plt.show()


# 3.4.3.3 Facets

anscombe_plot = sns.lmplot(x='x', y='y', data=anscombe,
                           fit_reg=False,
                           col='dataset', col_wrap=2)
plt.show()

# create a FacetGrid
facet = sns.FacetGrid(tips, col = 'time')
# for each value in time, plot a histogram of the bill
facet.map(sns.distplot, 'total_bill', rug = True)
plt.show()

# The individual facets need not be univariate plots, as seen in Figure 3.35.
facet = sns.FacetGrid(tips, col='day', hue='sex')
facet = facet.map(plt.scatter, 'total_bill', 'tip')
facet = facet.add_legend()
plt.show()

#Another thing you can do with facets is to have one variable be faceted
# on the x-axis, and another variable faceted on the y-axis.
# We accomplish this by passing a row parameter.

facet = sns.FacetGrid(tips, col='time', row='smoker', hue='sex')
facet.map(plt.scatter, 'total_bill', 'tip')
plt.show()

#If you do not want all of the hue elements to overlap
# (i.e., you want this behavior in scatterplots, but not violin plots),
# you can use the sns.factorplot function.

facet = sns.catplot(x='day', y='total_bill',
                       hue='sex', data = tips,
                       row = 'smoker', col='time',
                       kind='violin')
plt.show()

# 3.5 Pandas Objects

# 3.5.1 Histograms from within Pandas
fig, ax = plt.subplots()
ax = tips['total_bill'].plot.hist()
plt.show()

# with an alpha channel, so we can see channel transparency

fig, ax = plt.subplots()
ax = tips[['total_bill', 'tip']].plot.hist(alpha=0.5, bins = 20, ax = ax)
plt.show()

# 3.5.2 Density Plot

fig, ax = plt.subplots()
ax = tips['tip'].plot.kde()
plt.show()

# 3.5.3 Scatterplot

fig, ax = plt.subplots()
ax = tips.plot.scatter(x = 'total_bill', y = 'tip', ax = ax)
plt.show()

# 3.5.4 Hexbin Plot

fig, ax = plt. subplots()
ax = tips.plot.hexbin(x = 'total_bill', y = 'tip', ax = ax)
plt.show()

# 3.5.5 Boxplot
fig, ax = plt.subplots()
ax = tips.plot.box(ax = ax)
plt.show()

# 3.6 Seaborn Themes and Styles

#The seaborn plots shown in this chapter have all used the default plot styles.
# We can change the plot style with the sns.set_style function.
# Typically, this function is run just once at the top of your code;
# all subsequent plots will use the same style set.

# intial plot for comparison:
fig, ax = plt.subplots()
ax = sns.violinplot(x = 'time', y = 'total_bill',
                    hue = 'sex', data = tips, split = True)
plt.show()

# set style and plot
sns.set_style('whitegrid')
fig, ax = plt.subplots()
ax = sns.violinplot(x = 'time', y = 'total_bill',
                    hue = 'sex', data = tips,
                    split = True)
plt.show()

# the following code shows what all the styles look like:

fig = plt.figure()
seaborn_styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
for idx, style in enumerate(seaborn_styles):
    plot_position = idx+1
    with sns.axes_style(style):
        ax = fig.add_subplot(2, 3, plot_position)
        violin = sns.violinplot(x = 'time', y = 'total_bill',
                                data = tips, ax = ax)
fig.tight_layout()
plt.show()
