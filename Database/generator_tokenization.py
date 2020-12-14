# Imports
import glob
import os
import pandas as pd
import re
import spacy
import sys
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy_iwnlp import spaCyIWNLP


# Global Variables
META_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"
TEXT_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
SUMMARY_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\summary_files\\"

STOPWORDS = stopwords.words("english")
MODEL = "en_core_web_log"
EXPORT_PROGRESS = 0

'''
MODEL = "de_core_news_lg" # de_core_news_sm
LEMMATIZER = "C:\\Users\\didid\\GitHub-Respository\\AutomaticTextSummarization\\Database\\iwnlp_lemmatizer.json"
STOPWORDS = stopwords.words("german")
'''


# Methods
def read_text(file_path):
    with open(file_path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
        text = ". ".join(lines)
        f.close()

    return text


'''
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    # text = re.sub("[\(§\[§].*?[\)\]]", "", text)
    # text = re.sub("[^a-zA-Z]", "", text)

    return text


def lemmatize_text(text):
    global STOPWORDS
    global MODEL

    nlp = spacy.load(MODEL)
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc if token.lemma_ not in STOPWORDS])

    # nlp = spacy.load(MODEL)
    # nlp.add_pipe(spaCyIWNLP(lemmatizer_path=LEMMATIZER))
    # doc = nlp(text)
    # words = []

    # for token in doc:
    #     if token._.iwnlp_lemmas is not None:
    #         lemma = token._.iwnlp_lemmas[0]

    #         if lemma not in STOPWORDS:
    #             words.append(lemma)

    # text = " ".join(word for word in words)
    
    return text
'''


def export_text(file_path, doc):
    global EXPORT_PROGRESS

    try:
        with open(file_path, "w", encoding="utf8") as f:
            f.write(doc)
            f.close()

        EXPORT_PROGRESS += 1
        print_progress()

    except Exception as e:
        print(e)


def print_progress():
    global EXPORT_PROGRESS

    sys.stdout.write("\rFile %i..." % EXPORT_PROGRESS)
    sys.stdout.flush()


# Main
def main():
    global TEXT_PATH
    global META_PATH
    global TOKENIZED_TEXT_PATH
    global TOKENIZED_SUMMARY_PATH

    os.chdir(META_PATH)
    meta_files = glob.glob("*.csv")

    for file in meta_files:
        if "cnn_dailymail" in file or "wikihow" in file or "tldr" in file: # TODO: Select corpora
            df = pd.read_csv(file, index_col=0)

            for id in df["ID"]:
                text_file = TEXT_PATH + id + ".txt"
                tokenized_text_file = TOKENIZED_TEXT_PATH + id + ".txt"

                summary_file = SUMMARY_PATH + id + ".txt"
                tokenized_summary_file = TOKENIZED_SUMMARY_PATH + id + ".txt"

                if os.path.isfile(text_file) \
                        and os.path.isfile(summary_file) \
                        and not os.path.isfile(tokenized_text_file) \
                        and not os.path.isfile(tokenized_summary_file):
                    text = read_text(text_file)
                    summary = read_text(summary_file)

                    tokenized_text = word_tokenize(text)
                    tokenized_summary = word_tokenize(summary)

                    export_text(tokenized_text_file, tokenized_text)
                    export_text(tokenized_summary_file, tokenized_summary)


if __name__ == "__main__":
    main()
