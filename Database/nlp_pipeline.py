# Imports
import re
import spacy

from nltk.corpus import stopwords
from spacy_iwnlp import spaCyIWNLP


# Classes
class Pipeline:
    def __init__(self, raw_corpus, model, lemmatizer):
        self.raw_corpus = raw_corpus
        self.processed_corpus = []
        self.model = model
        self.lemmatizer = lemmatizer
        self.stopwords = stopwords.words("german")

    def process(self):
        nlp = spacy.load(self.model)
        nlp.add_pipe(spaCyIWNLP(lemmatizer_path=self.lemmatizer))

        for text in self.raw_corpus:
            doc = nlp(clean_text(text))
            tuples = []

            for token in doc:
                if token.lemma_ not in self.stopwords:
                    word = token.text
                    pos_tag = token.pos_
                    lemma = token._.iwnlp_lemmas[0] if token._.iwnlp_lemmas is not None else "TBD"

                    tuple = (word, pos_tag, lemma)
                    tuples.append(tuple)

            self.processed_corpus.append(tuples)
            print(tuples)

        print(self.processed_corpus)

        # TODO: Export lemmatized texts
        # TODO: Update return value according to the needed output


# Methods
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[\(§\[§].*?[\)\]]", "", text)

    return text
