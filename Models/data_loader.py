# Imports
import os
import glob
import pandas as pd
import random


# Methods
class DataLoader:
    def __init__(self, meta_path, text_path, summary_path, corpus_filter):
        self.meta_path = meta_path
        self.text_path = text_path
        self.summary_path = summary_path
        self.corpus_filter = corpus_filter

        self.meta_corpus = pd.DataFrame()
        self.corpus = []

        self.read_meta_files()
        self.setup_corpus()

    def read_meta_files(self):
        os.chdir(self.meta_path)
        files = glob.glob("*.csv")
        files = [self.meta_path + file for file in files]
        meta_files = []

        for corpus_name in self.corpus_filter:
            [meta_files.append(file) for file in files if os.path.isfile(file) and corpus_name in file]

        for file in meta_files:
            df = pd.read_csv(file, index_col=0)
            self.meta_corpus = pd.concat([self.meta_corpus, df], ignore_index=True)

    def setup_corpus(self):
        text_corpus = []
        summary_corpus = []

        for id in self.meta_corpus["ID"]:
            text_file = self.text_path + id + ".txt"
            summary_file = self.summary_path + id + ".txt"

            try:
                text = self.read_text(text_file)
                summary = self.read_text(summary_file)

                text_corpus.append(text)
                summary_corpus.append(summary)

            except Exception as e:
                print(e)

        self.corpus = list(zip(text_corpus, summary_corpus))

    def train_test_split(self, size, shuffle):
        train_data = []
        test_data = []

        if shuffle:
            random.shuffle(self.corpus)

        for pair in self.corpus:
            selector = random.random()
            train_data.append(pair) if selector < size else test_data.append(pair)

        X_train = [pair[0] for pair in train_data]
        y_train = [pair[1] for pair in train_data]

        X_test = [pair[0] for pair in test_data]
        y_test = [pair[1] for pair in test_data]

        return X_train, y_train, X_test, y_test

    @staticmethod
    def read_text(file_path):
        with open(file_path, "r", encoding="utf8", errors="ignore") as f:
            lines = f.readlines()
            text = "\n".join(lines)
            f.close()

        return text
