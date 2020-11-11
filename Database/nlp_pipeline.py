# Imports
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from german_lemmatizer import lemmatize


# Classes
class Pipeline:
    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_cleaned = []
        self.corpus_tokenized = []
        self.corpus_lemmatized = []

    def process(self):
        for text in self.corpus:
            c = Cleaner(text)
            c.Process()
            self.corpus_cleaned.append(c.text)

            t = Tokenizer(text)
            t.Process()
            self.corpus_tokenized.append(t.text)

            l = Lemmatizer(text)
            l.Process()
            self.corpus_lemmatized.append(l.text)


class Cleaner:
    def __init__(self, text):
        self.text = text

    def process(self):
        text = self.text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub("[\(§\[§].*?[\)\]]", "", text)

        self.text = text


class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.tokens = None

    def process(self):
        self.tokens = word_tokenize(self.text)


class Lemmatizer:
    def __init__(self, tokens):
        self.tokens = tokens
        self.words = None

    def process(self):
        words = self.tokens
        stop_words = stopwords.words("german")

        self.words = [lemmatize(word) for word in words if word not in stop_words]
