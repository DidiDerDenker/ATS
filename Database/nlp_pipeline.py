# Imports
import re
import spacy

from nltk.corpus import stopwords
from spacy_iwnlp import spaCyIWNLP


# Classes
class Pipeline:
    def __init__(self, raw_corpus, model, lemmatizer):
        self.raw_corpus = raw_corpus
        self.new_corpus = []
        self.model = model
        self.lemmatizer = lemmatizer
        self.stopwords = stopwords.words("german")

    def process(self):
        nlp = spacy.load(self.model)
        nlp.add_pipe(spaCyIWNLP(lemmatizer_path=self.lemmatizer))

        for doc in self.raw_corpus:
            doc = nlp(clean_text(doc))
            words = []

            for token in doc:
                if token.lemma_ not in self.stopwords:
                    # print(token.text, token.pos_, token._.iwnlp_lemmas
                    words.append(token.text)

            text = " ".join(word for word in words)
            self.new_corpus.append(text)

        # TODO: Export lemmatized texts
        # TODO: Update return value according to the needed output


# Methods
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[\(§\[§].*?[\)\]]", "", text)

    return text
