# Imports
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


# Global Variables
META_PATH = "C:\\Temp\\Corpus\\meta_files\\"
TEXT_PATH = "C:\\Temp\\Corpus\\text_files\\"
SUMMARY_PATH = "C:\\Temp\\Corpus\\summary_files\\"


# Methods
def read_text(file_path):
    with open(file_path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
        text = "\n".join(lines)
        f.close()

    return text


def data_loader(corpus_name):
    global META_PATH
    global TEXT_PATH
    global SUMMARY_PATH

    os.chdir(META_PATH)
    meta_files = glob.glob("*.csv")

    text_corpus = []
    summary_corpus = []

    for file in meta_files:
        if corpus_name in file:
            df = pd.read_csv(file, index_col=0)

            for id in df["ID"]:
                try:
                    text_file = TEXT_PATH + id + ".txt"
                    summary_file = SUMMARY_PATH + id + ".txt"

                    text = read_text(text_file)
                    summary = read_text(summary_file)

                    text_corpus.append(text)
                    summary_corpus.append(summary)

                except Exception as e:
                    print(e)

    # corpus = list(zip(text_corpus, summary_corpus))

    return text_corpus, summary_corpus


def get_sector_distribution():
    global META_PATH

    os.chdir(META_PATH)
    files = glob.glob("*.csv")

    distribution = {
        "cnn_dailymail": 0,
        "gigaword": 0,
        "open_legal": 0,
        "tldr": 0,
        "wikihow": 0,
        "wikipedia": 0
    }

    for file in files:
        df = pd.read_csv(file, index_col=0)
        size = len(df)

        distribution["cnn_dailymail"] += size if "cnn_dailymail" in file else 0
        distribution["gigaword"] += size if "gigaword" in file else 0
        distribution["open_legal"] += size if "open_legal" in file else 0
        distribution["tldr"] += size if "tldr" in file else 0
        distribution["wikihow"] += size if "wikihow" in file else 0
        distribution["wikipedia"] += size if "wikipedia" in file else 0

    distribution = {k: v for k, v in distribution.items() if int(v) != 0}
    n = sum(distribution.values())
    cnt = len(distribution.values())

    fig, ax = plt.subplots(figsize=[10, 6], tight_layout=True)
    plt.pie([float(v) for v in distribution.values()],
            labels=[str(k) for k in distribution.keys()],
            colors=plt.cm.magma(np.linspace(0.2, 0.8, cnt)),
            autopct="%1.1f%%", pctdistance=1.25, explode=[0.05] * cnt)

    plt.title("Sector distribution (" + str(n) + " files)", fontsize=14)
    fig.savefig("C:\\Temp\\Corpus\\Report\\sector_distribution.png")
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


def get_length_distribution(corpus, name):
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
    fig.savefig("C:\\Temp\\Corpus\\Report\\" + str(name) + "_length_distribution.png")
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
    fig.savefig("C:\\Temp\\Corpus\\Report\\n_gram_statistics_for_n_equals_" + str(n) + ".png")
    plt.clf()


# Main
def main():
    print("Reading corpus...")
    text_corpus, summary_corpus = data_loader("wikihow") # TODO: Select corpus

    print("Exporting sector distribution...")
    get_sector_distribution()

    print("Exporting text length distribution...")
    get_length_distribution(text_corpus, "text")
    get_length_distribution(summary_corpus, "summary")

    print("Exporting n-gram-statistics...")
    get_n_gram_statistics(text_corpus, 1)
    get_n_gram_statistics(text_corpus, 2)
    get_n_gram_statistics(text_corpus, 3)


if __name__ == "__main__":
    main()
