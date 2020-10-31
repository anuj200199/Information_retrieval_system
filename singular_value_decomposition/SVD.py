# This file implements Singular Value Decomposition on a predefined set of Documents and tries to find the similarity
# to a user Query.

# Importing the necessary libraries.
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import svd

# Making your own DataSet.
data = ["Human interface for Lab ABC computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user-perceived response time to error measurement",
        "The generation of random, binary, unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV: Widths of trees and well-quasi-ordering",
        "Graph minors: A survey"]

# Convert the dataset in a Count Array where the rows will be document index and columns as the vocabulary. Each cell
# contains the count of word corresponding to each document.
print("Converting documents to count Array")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
print("Vocabulary : " + str(vectorizer.get_feature_names()))  # This gives the feature names tht is the vocabulary.
count_Array = X.toarray().T

# Singular Value Decomposition.
U, S, V_transpose = svd(count_Array)

# populate Sigma with n x n diagonal matrix
Sigma = np.zeros((count_Array.shape[0], count_Array.shape[1]))
Sigma[:count_Array.shape[1], :count_Array.shape[1]] = np.diag(S)
print(Sigma.shape)

# Specify the number of dimensions to consider.
dimensions = int(input("Enter Dimensions: "))
Sigma = Sigma[:dimensions][:dimensions]
Sigma_inverse = np.linalg.inv(Sigma)
U = U[:, :9]

user_Query = input("Enter User Query: ")
user_Query = [user_Query]

# Count Array of User Query
query = vectorizer.transform(user_Query)
count_user_query = query.toarray()

# Find Similarity of User Query with Documents.
Query_k = np.linalg.multi_dot([count_user_query, U, Sigma_inverse])
similarity = cosine_similarity(np.transpose(V_transpose), Query_k)
sort_by_top_match = np.argmax(similarity, axis=0)
highest_similarity_document = data[sort_by_top_match[0]]
print("The highest similarity document id : " + highest_similarity_document)
