from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

X = pd.read_csv('data/aggregated_data.csv')
X = X.drop(columns=['srch_id'])
X = X.fillna(0)

X = np.array(X)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(X)
NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=5, p=2,
         radius=1.0)

distances, indices = knn.kneighbors(X)
print(distances)
print(indices)