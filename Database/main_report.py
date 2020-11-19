# Imports
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nlp_pipeline as nlp

from collections import Counter


# Global Variables
TEXT_FILES = "C:\\Temp\\Corpus\\"
META_FILES = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"
MODEL = "de_core_news_lg" # de_core_news_sm
LEMMATIZER = "C:\\Users\\didid\\GitHub-Respository\\AutomaticTextSummarization\\Database\\iwnlp_lemmatizer.json"


# Methods
def read_files():
    global TEXT_FILES
    global META_FILES

    os.chdir(META_FILES)
    meta_files = glob.glob("*.xlsx")
    text_files = []

    for file in meta_files:
        if "wikipedia" in file:
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
                text = nlp.clean_text(text)
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

    n = sum(distribution.values())

    fig, ax = plt.subplots(figsize=[10, 6], tight_layout=True)
    labels = distribution.keys()
    plt.pie(x=distribution, autopct="%.1f%%", explode=[0.05] * 2, labels=labels, pctdistance=0.5)
    plt.title("Sector distribution (" + n + " files)", fontsize=14)
    fig.savefig("./corpus_report/sector_distribution.png")
    plt.clf()


def get_text_length_distribution(corpus):
    distribution = []

    for text in corpus:
        size = len(text)
        distribution.append(size)

    fig = sns.distplot(distribution, hist=True, kde=False,
                       bins=int(max(distribution) / min(distribution)),
                       color="blue", hist_kws={"edgecolor": "black"})

    plt.title("Text length distribution")
    plt.xlabel("Length in words")
    plt.ylabel("Number of documents")
    fig.savefig("./corpus_report/text_length_distribution.png")
    plt.clf()


def get_word_distribution(corpus):
    vocabulary = {}

    for text in corpus:
        for word in text.split():
            if word in vocabulary:
                vocabulary[word] += 1

            else:
                vocabulary[word] = 1

    vocabulary = sorted(vocabulary.items(), key=lambda kv: (kv[1], kv[0]))

    items = vocabulary.items()
    distribution = list(items)[:20]
    distribution.plot() # TODO
    plt.show()
    plt.clf()


def get_n_gram_statistics(corpus, n):
    tokens = []

    for text in corpus:
        for token in text.split():
            if token != "":
                tokens.append(token)

    n_grams = zip(*[tokens[i:] for i in range(n)])
    n_grams = [" ".join(n_gram) for n_gram in n_grams]
    distribution = Counter(n_grams)

    # TODO: Export graphics
    print(distribution.most_common(20))


# Main
def main():
    global MODEL
    global LEMMATIZER

    print("Reading files...")
    files = read_files()

    print("Reading text...")
    raw_corpus = read_text(files)

    print("Processing nlp-pipeline...")
    nlp_pipeline = nlp.Pipeline(raw_corpus, MODEL, LEMMATIZER)
    nlp_pipeline.process()
    new_corpus = nlp_pipeline.new_corpus

    print("Exporting sector distribution...")
    get_sector_distribution()

    print("Exporting text length distribution...")
    get_text_length_distribution(new_corpus)

    print("Exporting word distribution...")
    get_word_distribution(new_corpus)

    print("Exporting n-gram-statistics...")
    get_n_gram_statistics(new_corpus, 2)
    get_n_gram_statistics(new_corpus, 3)
    get_n_gram_statistics(new_corpus, 5)


if __name__ == "__main__":
    main()
