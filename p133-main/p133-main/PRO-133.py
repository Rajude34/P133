import pandas as pd
import plotly.express as px

df = pd.read_csv("star_with_gravity.csv")

fig = px.scatter(df, x = "Radius", y = "Mass")
fig.show()

from typing import ValuesView
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

X = df.iloc[:, [0, 5]].values
print(X)

wcss = []

# here thee range is taken till eleven we just 10 cluster points
for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
  kmeans.fit(X)
# inertia method returns wcss for that model
  wcss.append(kmeans.inertia_)

# ploting a figure to show an elbow like structer in the graph
plt.figure(figsize = (10,5))
sns.lineplot(range(1,11), wcss, marker = 'o', color = "red")
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters = 3, init = "k-means++", random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize = (15,7))
sns.scatterplot(X[y_kmeans == 0, 0 ], X[y_kmeans == 0, 1], color = "yellow", label = "cluster 1")
sns.scatterplot(X[y_kmeans == 1, 0 ], X[y_kmeans == 1, 1], color = "blue", label = "cluster2")
sns.scatterplot(X[y_kmeans == 2, 0 ], X[y_kmeans == 2, 1], color = "green", label = "cluster3")
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],color = 'red', label = 'centroids', s = 100, marker = ',')
plt.grid(False)
plt.title("Cluster Of Planets")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.legend()
plt.show()