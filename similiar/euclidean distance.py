import numpy as np 
import pandas as pd
# uclidean_similarity

def uclidean_similarity(x, y):
    dis = np.sqrt(np.sum((x-y)**2))
    return 1 / (dis + 1e-5) 


doc1 = np.array((2, 3, 0, 1))
doc2 = np.array((1, 2, 3, 1))
doc3 = np.array((2, 1, 2, 2))
docQ = np.array((1, 1, 0, 1))


uclidean_similarity(doc1, docQ)
# 0.44721159550890216

uclidean_similarity(doc2, docQ)
# 0.31622676602000016

uclidean_similarity(doc3, docQ)
# 0.4082466238040005


# using package
li = [
    [2, 3, 0, 1],
    [1, 2, 3, 1],
    [2, 1, 2, 2],
    [1, 1, 0, 1]
]

df = pd.DataFrame(li)

from sklearn.metrics.pairwise import euclidean_distances
print(1 / (euclidean_distances(df, df) + 1e-5))


# cosine similarity 

from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity(df))
# [[1.         0.621059   0.66712438 0.9258201 ]
#  [0.621059   1.         0.85933785 0.59628479]
#  [0.66712438 0.85933785 1.         0.80064077]
#  [0.9258201  0.59628479 0.80064077 1.        ]]


# Jaccard Similarity

from sklearn.metrics.pairwise import paired_distances

jac_sim = 1 - paired_distances(np.array(df),np.array(df), metric = 'jaccard')
jac_sim

paired_distances()