# Imports
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


# Global Variables
META_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"
LEMMATIZATION_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\lemmatized_text_files\\"


# Methods
def read_text(file_path):
    with open(file_path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
        text = "\n".join(lines)
        f.close()

    return text


def data_loader(manual_filter):
    global META_PATH
    global LEMMATIZATION_PATH

    # TODO: What about summaries?

    os.chdir(META_PATH)
    meta_files = glob.glob("*.csv")
    corpus = []

    for file in meta_files:
        if manual_filter in file:
            df = pd.read_csv(file, index_col=0)

            for id in df["ID"]:
                try:
                    file_path = LEMMATIZATION_PATH + id + ".txt"
                    text = read_text(file_path)
                    corpus.append(text)

                except Exception as e:
                    print(e)

    return corpus


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
        df = pd.read_excel(file)
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
    '''
    "cnn_dailymail": Nicht anonymisierte Nachrichtenartikel
    "wikihow": Online-Wissensdatenbank als Antwort auf Texte
    "tldr": News aus Reddit-Threads
    '''

    print("Reading corpus...")
    corpus = data_loader("cnn_dailymail") # TODO: Select corpus

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
