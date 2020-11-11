# Imports
import glob
import os
import pandas as pd
import nlp_pipeline as nlp

from collections import Counter


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

                c = nlp.Cleaner(text)
                c.process()
                text = c.text

                corpus.append(text)

        except Exception as e:
            print(e)

    return corpus


def get_sector_distribution():
    global META_FILES

    os.chdir(META_FILES)
    files = glob.glob("*.xlsx")

    distribution = {"data_open_legal": 0,
                    "data_wikipedia": 0,
                    "data_enron": 0,
                    "data_mendeley": 0,
                    "data_uci": 0}

    for file in files:
        df = pd.read_excel(file)
        size = len(df)

        distribution["data_open_legal"] += size if "data_open_legal" in file else 0
        distribution["data_wikipedia"] += size if "data_wikipedia" in file else 0
        distribution["data_enron"] += size if "data_enron" in file else 0
        distribution["data_mendeley"] += size if "data_mendeley" in file else 0
        distribution["data_uci"] += size if "data_uci" in file else 0

    # TODO: Export graphics, add total text-file-count
    print(f"\tDistribution of sectors: {distribution}")


def get_text_length_distribution(corpus):
    distribution = {}

    for text in corpus:
        size = len(text)

        if size in distribution:
            distribution[size] += 1

        else:
            distribution[size] = 1

    distribution = sorted(distribution.items(), key=lambda kv: (kv[1], kv[0]))

    # TODO: Export graphic, search for stacked (value-grouped) charts
    print(f"\tDistribution of text lengths: {distribution}")


def get_word_distribution(corpus):
    for text in corpus:
        t = nlp.Tokenizer(text)
        t.process()
        tokens = t.tokens

        l = nlp.Lemmatizer(tokens)
        l.process()
        words = l.words

        print(tokens)
        print(words)
    exit()

    vocabulary = {}

    for text in corpus:
        tokens = text.split()

        for token in tokens:
            if token in vocabulary:
                vocabulary[token] += 1

            else:
                vocabulary[token] = 1

    vocabulary = sorted(vocabulary.items(), key=lambda kv: (kv[1], kv[0]))

    # TODO: Export graphics, head as well as tail items, after cleaning all texts
    print(f"\tDistribution of words: {vocabulary}")


def get_n_gram_statistics(corpus, n):
    tokens = []

    for text in corpus:
        for token in text.split():
            if token != "":
                tokens.append(token)

    n_grams = zip(*[tokens[i:] for i in range(n)])
    n_grams = [" ".join(n_gram) for n_gram in n_grams]
    distribution = Counter(n_grams)

    # TODO: Export graphics, head as well as tail items, after cleaning all texts
    print(distribution.most_common(20))


# Main
def main():
    print("Reading files...")
    files = read_files()

    print("Reading text...")
    corpus = read_text(files)

    print("Exporting sector distribution...")
    # get_sector_distribution()

    print("Exporting text length distribution...")
    # get_text_length_distribution(corpus)

    print("Exporting word distribution...")
    get_word_distribution(corpus)

    print("Exporting n-gram-statistics...")
    get_n_gram_statistics(corpus, 2)
    get_n_gram_statistics(corpus, 3)
    get_n_gram_statistics(corpus, 5)


if __name__ == "__main__":
    main()
