# Imports
import glob
import os
import pandas as pd
import re
import string
import spacy
import sys

from nltk.corpus import stopwords
from spacy_iwnlp import spaCyIWNLP


# Global Variables
TEXT_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
META_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"
LEMMATIZATION_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\lemmatized_files\\"

MODEL = "de_core_news_lg" # de_core_news_sm
LEMMATIZER = "C:\\Users\\didid\\GitHub-Respository\\AutomaticTextSummarization\\Database\\iwnlp_lemmatizer.json"
STOPWORDS = stopwords.words("german")
EXPORT_PROGRESS = 0


# Methods
def read_text(file_path):
    with open(file_path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
        text = "\n".join(lines)
        f.close()

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(None, string.punctuation)
    # text = re.sub("[\(§\[§].*?[\)\]]", "", text)

    return text


def lemmatize_text(text):
    global MODEL
    global LEMMATIZER
    global STOPWORDS

    nlp = spacy.load(MODEL)
    nlp.add_pipe(spaCyIWNLP(lemmatizer_path=LEMMATIZER))
    doc = nlp(text)
    words = []

    for token in doc:
        if token._.iwnlp_lemmas is not None:
            lemma = token._.iwnlp_lemmas[0]

            if lemma not in STOPWORDS:
                words.append(lemma)

    text = " ".join(word for word in words)

    return text


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
    global LEMMATIZATION_PATH
    global EXPORT_PROGRESS

    os.chdir(META_PATH)
    meta_files = glob.glob("*.xlsx")

    for file in meta_files:
        if "tensorflow" in file:
            df = pd.read_excel(file)

            for id in df["ID"]:
                input_path = TEXT_PATH + id + ".txt"
                output_path = LEMMATIZATION_PATH + id + ".txt"

                if not os.path.isfile(output_path):
                    text = read_text(input_path)
                    text = clean_text(text)
                    doc = lemmatize_text(text)
                    export_text(output_path, doc)

                else:
                    EXPORT_PROGRESS += 1
                    print_progress()


if __name__ == "__main__":
    main()
