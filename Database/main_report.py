# Imports
import glob
import os
import pandas as pd
import collections


# Global Variables
TEXT_FILES = "C:\\Temp\\Corpus\\"
META_FILES = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"


# Methods
def read_files():
    global TEXT_FILES
    global META_FILES

    os.chdir(META_FILES)
    meta_files = glob.glob("*.xlsx")
    text_files = []

    for file in meta_files:
        df = pd.read_excel(file)

        for id in df["ID"]:
            text_files.append(TEXT_FILES + id + ".txt")

    print(f"\tNumber of meta-files: {len(meta_files)}")
    print(f"\tNumber of text-files: {len(text_files)}")

    return text_files


def read_text(files):
    corpus = []

    for file in files:
        try:
            with open(file, "r", encoding="utf8", errors="ignore") as f:
                lines = f.readlines()
                text = "\n".join(lines)
                text = clean_text(text)
                corpus.append(text)

        except Exception as e:
            print(e)

    return corpus


def clean_text(text):
    # TODO: Check and clean texts properly

    '''
    text = text.str.lower()
    text = text.str.replace(r"@", "")
    text = text.str.replace(r"[^A-Za-z0-9öäüÖÄÜß()!?]", " ")
    text = text.str.replace("\s{2,}", " ")
    '''

    return text


def get_sector_distribution():
    global META_FILES

    os.chdir(META_FILES)
    files = glob.glob("*.xlsx")

    distribution = {"data_open_legal": 0,
                    "data_wikipedia": 0,
                    "data_corpus": 0,
                    "data_enron": 0,
                    "data_mendeley": 0}

    for file in files:
        df = pd.read_excel(file)
        size = len(df)

        distribution["data_open_legal"] += size if "data_open_legal" in file else 0
        distribution["data_wikipedia"] += size if "data_wikipedia" in file else 0
        distribution["data_corpus"] += size if "data_corpus" in file else 0
        distribution["data_enron"] += size if "data_enron" in file else 0
        distribution["data_mendeley"] += size if "data_mendeley" in file else 0

    # TODO: Export graphics, add total text-file-count
    print(f"\tDistribution of sectors: {distribution}")


def get_text_length_distribution(corpus):
    distribution = {}

    for file in corpus:
        size = len(file)

        if size in distribution:
            distribution[size] += 1

        else:
            distribution[size] = 1

    distribution = sorted(distribution.items(), key=lambda kv: (kv[1], kv[0]))

    # TODO: Export graphic, search for stacked (value-grouped) charts
    print(f"\tDistribution of text lengths: {distribution}")


def get_words(corpus):
    words = []

    # TODO: Extract all words in a non-distinct style from all documents and append each word to the list

    return words


def get_word_distribution(corpus):
    words = get_words(corpus)
    mcw = collections.Counter(words).most_common(100)

    # TODO: Setup a vocabulary, export graphics
    print(f"\tDistribution of words: {mcw}")


def get_n_grams(corpus, n):
    distribution = {}

    # TODO: Extract all n-grams in the corpus, export head and tail as a graphic
    print(f"\t: {distribution}")


# Main
def main():
    print("Reading files...")
    files = read_files()

    print("Reading text...")
    corpus = read_text(files)

    print("Exporting sector distribution...")
    get_sector_distribution()

    print("Exporting text length distribution...")
    get_text_length_distribution(corpus)

    print("Exporting word distribution...")
    # get_word_distribution(corpus)

    print("Exporting n-gram-statistics...")
    # get_n_grams(corpus, 3)


if __name__ == "__main__":
    main()
