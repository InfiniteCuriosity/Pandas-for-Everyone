# 16. Clustering! :)

# 16.2 k-Means clustering

##### -- import -- #####
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
pd.set_option('max_columns', None)
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

#########################

wine = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/wine.csv')
print(wine.head)

# We will drop the Cultivar column, since it correlates too closely with the actual
# clusters in our data.

wine = wine.drop('Cultivar', axis=1)
wine.head()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42).fit(wine.values)
kmeans = KMeans(n_clusters=3, random_state=42).fit(wine.values)

print(kmeans) # this does not look correct to me

print(np.unique(kmeans.labels_, return_counts=True))

# We can turn these labels into a dataframe that we can then add to our data set.

kmeans_3 = pd.DataFrame(kmeans.labels_, columns = ['cluster'])
print(kmeans_3.head())

# 16.2.1 Dimension Reduction with PCA

pca = PCA(n_components=2).fit(wine)

pca_trans = pca.transform(wine)

# give our projections a name
pca_trans_df = pd.DataFrame(pca_trans, columns=['pca1', 'pca2'])

# concatenate our data

kmeans_3 = pd.concat([kmeans_3, pca_trans_df], axis = 1)

print(kmeans_3.head())

# display our results :)

fig = sns.lmplot(x = 'pca1', y = 'pca2', data = kmeans_3, hue = 'cluster', fit_reg = False)
plt.show()

# 16.3 Hierarchial Clustering

wine = pd.read_csv('/Users/russellconte/Documents/Pandas for Everyone/pandas_for_everyone-master/data/wine.csv')
wine = wine.drop('Cultivar', axis = 1)

# 16.3.1 Complete Clustering

wine_complete = hierarchy.complete(wine)
fit = plt.figure()
dn = hierarchy.dendrogram(wine_complete)
plt.show() # this is pixelated, and not easy to see at higher magnfications. Can it be output as a vector file?

# 16.3.2 Single Clustering

wine_single = hierarchy.single(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_single)
plt.show()

# 16.3.3 Average Clustering
wine_average = hierarchy.average(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_average)
plt.show()

# 16.3.4 Centroid Clustering
wine_centroid = hierarchy.centroid(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(wine_centroid)
plt.show()

# 16.3.5 Manually setting the threshold

wine_complete = hierarchy.complete(wine)
fig = plt.figure()
dn = hierarchy.dendrogram(
    wine_complete,
    color_threshold=0.7 * max(wine_complete[:,2]),
    above_threshold_color='y')
plt.show()