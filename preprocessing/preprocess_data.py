from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer


def tokenize(sentence, tokenizer=' '):
    return sentence.split(tokenizer)


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
