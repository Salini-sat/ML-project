from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("C:\\Users\\KIIT0001\\Desktop\\ml\\income.csv")


scaler = MinMaxScaler()
df[['Age','Income($)']] = scaler.fit_transform(df[['Age','Income($)']])


km = KMeans(n_clusters=3, n_init=10, random_state=42)
df['cluster'] = km.fit_predict(df[['Age','Income($)']])


df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]


plt.scatter(df1.Age, df1['Income($)'], label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], label='Cluster 2')


plt.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    color='purple',
    marker='*',
    s=20,
    label='Centroids'
)

plt.xlabel("Age (scaled)")
plt.ylabel("Income (scaled)")
plt.legend()
plt.show()

# elbow method 
sse = []
for k in range(1,10):
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

plt.plot(range(1,10), sse)
plt.xlabel("K")
plt.ylabel("SSE")
plt.show()

df.to_csv("clustered_income.csv", index=False)
