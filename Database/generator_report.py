# Imports
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_loader_corpus as dlc

from collections import Counter


# Global Variables
META_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"
LEMMATIZATION_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\lemmatized_files\\"


# Methods
def get_sector_distribution():
    global META_PATH

    os.chdir(META_PATH)
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


def get_stacked_distribution(values, n):
    v_min = 1000.0
    v_max = 10000.0
    step_size = (v_max - v_min) / (n - 1)
    borders = np.linspace(v_min, v_max, num=n)
    parts = [str(b)[:-2] + "-" + str(b + step_size - 1)[:-2] for b in borders][:-1]
    distribution = {}

    for part in parts:
        distribution[str(part)] = 0

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

    n = 25
    distribution = get_stacked_distribution(values, n)

    fig, ax = plt.subplots(figsize=[14, 6], tight_layout=True)
    plt.bar(distribution.keys(), distribution.values(),
            color=plt.cm.magma(np.linspace(0.8, 0.2, n)))

    plt.title("Text length distribution")
    plt.xlabel("Length in words")
    plt.ylabel("Number of documents")
    plt.xticks(rotation=30)
    fig.savefig("C:\\Temp/corpus_report/text_length_distribution.png")
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
    global META_PATH
    global LEMMATIZATION_PATH

    print("Reading corpus...")
    corpus = dlc.main(META_PATH, LEMMATIZATION_PATH)

    print("Exporting sector distribution...")
    get_sector_distribution()

    print("Exporting text length distribution...")
    get_text_length_distribution(corpus)

    print("Exporting n-gram-statistics...")
    get_n_gram_statistics(corpus, 1)
    get_n_gram_statistics(corpus, 2)
    get_n_gram_statistics(corpus, 3)


if __name__ == "__main__":
    main()
