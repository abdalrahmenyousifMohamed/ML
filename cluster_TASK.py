import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
#nms = sns.get_dataset_names()
data = sns.load_dataset('car_crashes')
if data.empty:data.interpolate(inplace=True)
print(data) # speeding  alcohol
plt.scatter(data.speeding,data.alcohol)
plt.xlabel("speeding")
plt.ylabel('alcohol')
x= data.iloc[:,1:3]
print(x)
kmeans = KMeans(n_clusters=3)
cluster_table = kmeans.fit_predict(x)
print(cluster_table)
data["cluster_labels"]=cluster_table
l = kmeans.cluster_centers_ # centroid 3 vals
print("centered values {}".format(l))
print("labels is {}".format(kmeans.labels_))
print("iterations is {}".format(kmeans.n_iter_))
group0 = data[data.cluster_labels==0]
group1 = data[data.cluster_labels==1]
group2 = data[data.cluster_labels==2]
plt.scatter(group0.speeding,group0.alcohol,label="cluster 0")
plt.scatter(group1.speeding,group1.alcohol,label="cluster 1")
plt.scatter(group2.speeding,group2.alcohol,label="cluster 2")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],label="centroid",marker = "*")
plt.legend()
plt.show()