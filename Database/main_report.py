# Imports
import glob
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                    "data_mendeley": 0,
                    "data_uci": 0}

    for file in files:
        df = pd.read_excel(file)
        size = len(df)

        distribution["data_open_legal"] += size if "data_open_legal" in file else 0
        distribution["data_wikipedia"] += size if "data_wikipedia" in file else 0
        distribution["data_mendeley"] += size if "data_mendeley" in file else 0
        distribution["data_uci"] += size if "data_uci" in file else 0

    distribution = {k: v for k, v in distribution.items() if int(v) != 0}
    n = sum(distribution.values())
    cnt = len(distribution.values())

    fig, ax = plt.subplots(figsize=[10, 6], tight_layout=True)
    plt.pie([float(v) for v in distribution.values()],
            labels=[str(k) for k in distribution.keys()],
            colors=plt.cm.magma(np.linspace(0.2, 0.8, cnt)),
            autopct="%1.1f%%", pctdistance=1.25, explode=[0.05] * cnt)

    plt.title("Sector distribution (" + str(n) + " files)", fontsize=14)
    fig.savefig("C:\\Temp/corpus_report/sector_distribution.png")
    plt.clf()


def get_stacked_distribution(values, n_parts):
    v_min = min(values)
    v_max = max(values)
    n_size = math.floor((v_max - v_min) / (n_parts - 1))
    parts = []
    distribution = {}

    for i in range(0, n_parts - 1):
        n = v_min + i * n_size
        parts.append(n)

    for part in parts:
        distribution[str(part) + "-" + str(part + n_size - 1)] = 0

    for v in values:
        for d in distribution.keys():
            start = int(d.split("-")[0])
            end = int(d.split("-")[1])

            if start <= v <= end:
                distribution[d] += 1

    return distribution


def get_text_length_distribution(corpus):
    values = []

    for text in corpus:
        size = len(str(text))
        values.append(size)

    n_parts = 20
    distribution = get_stacked_distribution(values, n_parts)

    # TODO: Display bars correctly, e.g. try another plot, maybe barh as well
    print(distribution)
    fig, ax = plt.subplots(figsize=[14, 6], tight_layout=True)
    plt.bar([int(k.split("-")[0]) for k in distribution.keys()], [int(v) for v in distribution.values()],
            color=plt.cm.magma(np.linspace(0.8, 0.2, n_parts)),
            tick_label=[str(k) for k in distribution.keys()], width=0.8)

    plt.title("Text length distribution")
    plt.xlabel("Length in words")
    plt.ylabel("Number of documents")
    plt.xticks(rotation=30)
    fig.savefig("C:\\Temp/corpus_report/text_length_distribution.png")
    plt.clf()


def get_word_distribution(corpus):
    vocabulary = {}

    for text in corpus:
        for word in text.split():
            if word in vocabulary:
                vocabulary[word] += 1

            else:
                vocabulary[word] = 1

    n_top = 20
    distribution = {k: v for k, v in sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)}
    distribution = list(distribution.items())[:n_top]

    words = [str(x[0]) for x in distribution]
    counts = [int(x[1]) for x in distribution]
    y_pos = np.arange(len(words))

    fig, ax = plt.subplots(figsize=[10, 6], tight_layout=True)
    plt.title("Word distribution")
    plt.barh(y_pos, counts, align="center", color=plt.cm.magma(np.linspace(0.2, 0.8, n_top)))
    plt.xlabel("Counts")
    plt.yticks(y_pos, labels=words)
    plt.gca().invert_yaxis()
    fig.savefig("C:\\Temp/corpus_report/word_distribution.png")
    plt.clf()


def get_n_gram_statistics(corpus, n):
    tokens = []

    for text in corpus:
        for token in text.split():
            if token != "":
                tokens.append(token)

    n_top = 20
    n_grams = zip(*[tokens[i:] for i in range(n)])
    n_grams = [" ".join(n_gram) for n_gram in n_grams]
    distribution = Counter(n_grams).most_common(n_top)

    n_grams = [str(v[0]) for v in distribution]
    counts = [int(v[1]) for v in distribution]
    y_pos = np.arange(len(n_grams))

    fig, ax = plt.subplots(figsize=[10, 6], tight_layout=True)
    plt.title("N-Gram statistics (n=" + str(n) + ")")
    plt.barh(y_pos, counts, align="center", color=plt.cm.magma(np.linspace(0.2, 0.8, n_top)))
    plt.xlabel("Counts")
    plt.yticks(y_pos, labels=n_grams)
    plt.gca().invert_yaxis()
    fig.savefig("C:\\Temp/corpus_report/n_gram_statistics_for_n_equals_" + str(n) + ".png")
    plt.clf()


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
