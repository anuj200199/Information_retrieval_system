from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def tokenize(sentence, tokenize_char=' '):
    return sentence.split(tokenize_char)


def reg_exp_tokenizer(sentence, tokenize_char=' '):
    tokenizer = RegexpTokenizer(tokenize_char)
    return tokenizer.tokenize(sentence)


def remove_stop_words(words):
    stopword = list(set(stopwords.words('english')))
    return [word for word in words if word.lower() not in stopword]


def stem_mapper(stemmer_type):
    stemmer = {'Poster Stemmer': PorterStemmer(), 'Snowball Stemmer': SnowballStemmer(language='english')}
    if stemmer_type in stemmer.keys():
        return stemmer[stemmer_type]
    else:
        return PorterStemmer()


def stemming(word, stemmer='Porter Stemmer'):
    stemmer = stem_mapper(stemmer)
    return stemmer.stem(word)


def list_word_stemmer(words, stemmer='Porter Stemmer'):
    stemmed_string = ''
    for word in words:
        stemmed_word = stemming(word, stemmer)
        stemmed_string = stemmed_string + ' ' + stemmed_word
    return stemmed_string


def label_encoding(label_encode_target):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(label_encode_target)


def split_into_train_and_test(input_data, target_data, test_size=0.2, random_state=1):
    return train_test_split(input_data, target_data, test_size=test_size, random_state=random_state)
