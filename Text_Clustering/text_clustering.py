# This script implements unsupervised learning for identifying similar documents using k-Means Clustering . Identify
# optimal value of K using elbow method


# Importing the libraries.
import re
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import random as rd
import numpy as np
from preprocessing.preprocess_data import list_word_stemmer, remove_stop_words, tokenize

# Create dataset for Clustering.
docs = ["This little kitty came to play when I was eating at a restaurant.",
        "Merley has the best squooshy kitten belly.",
        "Google Translate app is incredible.",
        "If you open 100 tab in google you get a smiley face.",
        "Best cat photo I've ever taken.",
        "Climbing ninja cat.",
        "Impressed with google map feedback.",
        "Key promoter extension for Google Chrome."]


# Preprocess the Text
preprocessed_docs = []
for doc in docs:
    text = re.sub('[^a-zA-Z]', ' ', doc)
    text = tokenize(text)
    text = remove_stop_words(text)
    text = list_word_stemmer(text)
    preprocessed_docs.append(text)

print("Documents After preprocessing : ")
print(preprocessed_docs)

# Fitting TF-IDF vectorizer to preprocessed documents
tfidf_documents = TfidfVectorizer()
tfidf_documents_array = tfidf_documents.fit_transform(preprocessed_docs).toarray()

# Find optimal k using elbow method.
wcss = []
for i in range(1, 6):
    k_means = KMeans(n_clusters=i, init='k-means++', random_state=1)
    k_means.fit(tfidf_documents_array)
    wcss.append(k_means.inertia_)
plt.plot(range(1, 6), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Optimal Cluster for this dataset is 2. Fitting K-Means to the dataset
k_means = KMeans(n_clusters=2, init='k-means++', random_state=42)
clustered_documents = k_means.fit_predict(tfidf_documents_array)
print("Results After Clustering Using sklearn K Means : ")
print(clustered_documents)

print("Start Clustering using Own Logic")
# Clustering Documents by writing your own logic.
m = tfidf_documents_array.shape[0]
n = tfidf_documents_array.shape[1]
iterations = 5  # No. Of Iterations
K = 2  # Clusters

# Randomly select points equal to number of clusters.
centroids_index = []
for i in range(K):
    rand = rd.randint(0, m - 1)
    centroids_index.append(rand)

# Put centroid value.
centroids = []
for i in range(0, K):
    centroids.append(tfidf_documents_array[centroids_index[i]][:])

# Iterate over iterations and also view the clustering per iterations.
dist1 = []
dist2 = []
for iteration in range(0, iterations):
    for i in range(0, len(tfidf_documents_array)):
        dist_c1 = float(sqrt(np.sum((tfidf_documents_array[i] - centroids[0]) ** 2)))
        dist_c2 = float(sqrt(np.sum((tfidf_documents_array[i] - centroids[1]) ** 2)))
        dist1.append(dist_c1)
        dist2.append(dist_c2)

    # New centroid.
    c1 = []
    c2 = []
    new_points1 = []
    new_points2 = []
    for j in range(0, len(tfidf_documents_array)):
        if dist1[j] > dist2[j]:
            c1.append(j)
            new_points1.append(list(tfidf_documents_array[j]))
        else:
            c2.append(j)
            new_points2.append(list(tfidf_documents_array[j]))

    centroids[0] = np.mean(new_points1, axis=0)
    centroids[1] = np.mean(new_points2, axis=0)
    print(c1, c2)

print("Final Results after Clustering using Own Logic : ")
print("Documents in First Cluster: ")
print(c1)
print("Documents in Second CLuster: ")
print(c2)
