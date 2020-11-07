# We will include the gutenberg data for implementing TF-IDF Vectorizer.

# Importing the necessary libraries
from nltk.corpus import gutenberg
import numpy as np
import preprocessing.preprocess_data as preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Reading the Gutenberg Database")
gutenberg_file_ids = gutenberg.fileids()
print("Total Number of Documents are " + str(len(gutenberg_file_ids)))

# Get the words for each fileId
file_words = []
for file_id in gutenberg_file_ids:
    file_words.append(gutenberg.words(file_id))

# Preprocessing Steps
print("Start Preprocessing Steps")
# 1. Remove Stop Words from each File
for i in range(len(file_words)):
    file_words[i] = preprocess.remove_stop_words(file_words[i])

# 2. Stem the words based on your preference. I will be using Porter Stemmer.
stemmed_words = []
for words in file_words:
    stemmed_string = preprocess.list_word_stemmer(words, 'Porter Stemmer')
    stemmed_words.append(stemmed_string)
print("End preprocessing Steps")

# Start TF-IDF.
tf_idf = TfidfVectorizer()
X_data = tf_idf.fit_transform(stemmed_words)

# To find similarity with the user Query
user_query = str(input("Enter User Query :  "))  # Example : amusement occupation sensations.

input_words = preprocess.tokenize(user_query, ' ')
input_words = preprocess.remove_stop_words(input_words)
input_words = preprocess.list_word_stemmer(input_words)

X_user_query = tf_idf.transform(input_words.split(' '))

# Find Cosine Similarity between user_query and Documents
cosine_sim = cosine_similarity(X_data, X_user_query, dense_output=True)

# Find top matching documents
top_document_index = np.argmax(cosine_sim, axis=0)

print("The top matching document for the user Query: " + user_query + " is indexed at " + str(top_document_index[0]))
