# Imports
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from germalemma import GermaLemma


# Classes
'''
Input: Corpus that contains multiple text-objects
Task: Preprocess corpus
Output: Corpus that contains preprocessed text-objects and different lists
'''
class Pipeline:
    def __init__(self, corpus):
        self.raw_corpus = corpus
        self.corpus_cleaned = []
        self.corpus_tokenized = []
        self.corpus_tagged = []
        self.corpus_lemmatized = []

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

    def export(self):
        # TODO: Export all properties into a given output-directory in order to read them later
        exit()


'''
Input: Text-object
Task: Clean text
Output: Text-object
'''
class Cleaner:
    def __init__(self, text):
        self.text = text

    def process(self):
        text = self.text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub("[\(§\[§].*?[\)\]]", "", text)

        self.text = text


'''
Input: Text-object
Task: Tokenize text
Output: Word-tokens
'''
class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.tokens = None

    def process(self):
        self.tokens = word_tokenize(self.text)


'''
Input: Word-tokens
Task: POS-Tag tokens
Output: Tuples of tokens and pos-tags
'''
class Tagger:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos_tags = None
        self.tuples = []

    def process(self):
        self.tuples = nltk.pos_tag(self.tokens)  # TODO: Optimize tagger, search for german pos-tagging, don't ignore numbers


'''
Input: Tuples of tokens and pos-tags
Task: Lemmatize tokens
Output: Word-list, text-object
'''
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
                lemma = self.lemmatizer.find_lemma(word, pos_tag)  # TODO: Optimize lemmatizer, search for good german lemmatizers that recognizes numbers
                self.words.append(lemma.lower())

            except Exception as e:
                print(e)
