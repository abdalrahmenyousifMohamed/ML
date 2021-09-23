import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
#nms = sns.get_dataset_names()
data = sns.load_dataset('iris')
print(data)
plt.scatter(data.sepal_length,data.sepal_width)
plt.xlabel("sepal_length")
plt.ylabel('sepal_width')
x= data.iloc[:,:2]
kmeans = KMeans(n_clusters=3)
cluster_table = kmeans.fit_predict(x)
print(cluster_table)
data["cluster_labels"]=cluster_table
l = kmeans.cluster_centers_ # centroid 3 vals
kmeans.labels_
kmeans.n_iter_
group0 = data[data.cluster_labels==0]
group1 = data[data.cluster_labels==1]
group2 = data[data.cluster_labels==2]
plt.scatter(group0.sepal_length,group0.sepal_width,label="cluster 0")
plt.scatter(group1.sepal_length,group1.sepal_width,label="cluster 1")
plt.scatter(group2.sepal_length,group2.sepal_width,label="cluster 2")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],label="centroid",marker = "*")
plt.legend()
plt.show()