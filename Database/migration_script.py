# Imports
import os
import glob
import pandas as pd

from shutil import copyfile


# Global Variables
META_PATH = "C:\\Temp\\Corpus\\meta_files\\"
TEXT_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
SUMMARY_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\summary_files\\"
NEW_TEXT_PATH = "C:\\Temp\\Corpus\\text_files\\"
NEW_SUMMARY_PATH = "C:\\Temp\\Corpus\\summary_files\\"


# Methods
def read_meta_files(corpus_filter):
    global META_PATH

    os.chdir(META_PATH)
    files = glob.glob("*.csv")
    files = [META_PATH + file for file in files]

    meta_files = []
    meta_corpus = pd.DataFrame()

    for corpus_name in corpus_filter:
        [meta_files.append(file) for file in files if os.path.isfile(file) and corpus_name in file]

    for file in meta_files:
        df = pd.read_csv(file, index_col=0)
        meta_corpus = pd.concat([meta_corpus, df], ignore_index=True)

    return meta_corpus


def copy_files(meta_corpus):
    global TEXT_PATH
    global SUMMARY_PATH
    global NEW_TEXT_PATH
    global NEW_SUMMARY_PATH

    n_iter = 0
    n_load = len(meta_corpus["ID"])

    for id in meta_corpus["ID"]:
        text_file = TEXT_PATH + id + ".txt"
        summary_file = SUMMARY_PATH + id + ".txt"

        new_text_file = NEW_TEXT_PATH + id + ".txt"
        new_summary_file = NEW_SUMMARY_PATH + id + ".txt"

        try:
            copyfile(text_file, new_text_file)
            copyfile(summary_file, new_summary_file)

            n_iter += 1
            print(f"File {n_iter}/{n_load} | {(n_iter / n_load):.5f}%")

        except Exception as e:
            print(e)

    print("Migration done...")


# Main
def main():
    corpus_filter = ["cnn_dailymail", "wikihow", "tldr"]
    meta_corpus = read_meta_files(corpus_filter)
    copy_files(meta_corpus)


if __name__ == "__main__":
    main()
