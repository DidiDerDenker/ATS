# Imports
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from germalemma import GermaLemma


# Classes
class Pipeline:
    def __init__(self, corpus):
        self.raw_corpus = corpus
        self.corpus_cleaned = []
        self.corpus_tokenized = []
        self.corpus_tagged = []
        self.corpus_lemmatized = []
        self.text_corpus = []

    def process(self):
        for text in self.raw_corpus:
            c = Cleaner(text)
            c.process()
            self.corpus_cleaned.append(c.text)

            t = Tokenizer(c.text)
            t.process()
            self.corpus_tokenized.append(t.tokens)

            p = Tagger(t.tokens)
            p.process()
            self.corpus_tagged.append(p.tuples)

            l = Lemmatizer(p.tuples)
            l.process()
            self.corpus_lemmatized.append(l.words)

            self.text_corpus.append(l.text)


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


class Tagger:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos_tags = None
        self.tuples = []

    def process(self):
        self.tuples = nltk.pos_tag(self.tokens)  # TODO: Optimize tagger


class Lemmatizer:
    def __init__(self, tuples):
        self.stopwords = stopwords.words("german")
        self.lemmatizer = GermaLemma()
        self.tuples = tuples
        self.words = []
        self.text = None

    def process(self):
        for tuple in self.tuples:
            word = tuple[0]
            pos_tag = tuple[1]

            try:
                lemma = self.lemmatizer.find_lemma(word, pos_tag)  # TODO: Optimize lemmatizer
                self.words.append(lemma.lower())

            except Exception as e:
                print(e)

        self.text = " ".join(self.words)
