# Imports
import glob
import os
import pandas as pd
import re
import spacy
import sys

from nltk.corpus import stopwords
from spacy_iwnlp import spaCyIWNLP


# Global Variables
TEXT_FILES = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
META_FILES = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"
LEMMATIZED_FILES = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\lemmatized_files\\"

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
    text = re.sub("[\(§\[§].*?[\)\]]", "", text)

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

        sys.stdout.write("\rFile %i..." % (EXPORT_PROGRESS + 1))
        sys.stdout.flush()
        EXPORT_PROGRESS += 1

    except Exception as e:
        print(e)


# Main
def main():
    global TEXT_FILES
    global META_FILES

    os.chdir(META_FILES)
    meta_files = glob.glob("*.xlsx")

    for file in meta_files:
        if "open_legal" not in file:
            df = pd.read_excel(file)

            for id in df["ID"]:
                input_path = TEXT_FILES + id + ".txt"
                output_path = LEMMATIZED_FILES + id + ".txt"
                text = read_text(input_path)
                text = clean_text(text)
                doc = lemmatize_text(text)

                if len(doc) > 500:
                    export_text(output_path, doc)


if __name__ == "__main__":
    main()
