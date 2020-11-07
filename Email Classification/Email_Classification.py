# This script takes input data containing email and their associated labels as part of a Supervised learning task of
# classification.

# Importing the libraries.
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing.preprocess_data import reg_exp_tokenizer, remove_stop_words, list_word_stemmer, label_encoding, \
    split_into_train_and_test

# Read the data from the CSV File.
data = pd.read_csv('SPAM text message 20170820 - Data.csv')
print("Data Description : ")
print(data.describe())

# Preprocess the data.
for i in range(0, data.shape[0]):
    words = reg_exp_tokenizer(data.iloc[i][1], r'\w+')
    words = remove_stop_words(words)
    words = list_word_stemmer(words)
    words = words.translate(str.maketrans('', '', string.digits))
    data.iloc[i][1] = words

# Segregate the target fields from input fields.
emails = data.iloc[:, 1]
actual_class = data.iloc[:, 0]
encoded_actual_class = label_encoding(actual_class)

# Covert to TF-IDF vectors. One can also user Only Tf or Boolean model.
tf = TfidfVectorizer()
emails_tf_idf = tf.fit_transform(emails)
emails_tf_idf_array = emails_tf_idf.toarray()

# Split into train and test Data.
email_train_data, email_test_data, label_train_data, label_test_data = split_into_train_and_test(emails_tf_idf_array,
                                                                                                 encoded_actual_class,
                                                                                                 test_size=0.2,
                                                                                                 random_state=1)

# We will fit the TF-IDF vector using Gaussian Classification.
gb_classification = GaussianNB()
gb_classification.fit(email_train_data, label_train_data)

# This returns the output for each class in terms of probability.
predicted_test_data_prob = gb_classification.predict_proba(email_test_data)
# This returns only the class for which data belongs to max probability.
predicted_test_data = gb_classification.predict(email_test_data)

# Print the output i terms of confusion matrix and accuracy.
cm = confusion_matrix(label_test_data, predicted_test_data)
acc = accuracy_score(label_test_data, predicted_test_data)
print("Confusion Matrix: ")
print(cm)

print("Accuracy for the Model : " + str(acc))
