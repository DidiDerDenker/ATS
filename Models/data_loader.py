# Imports
import os
import glob
import pandas as pd

from nltk import word_tokenize


# Methods
class DataLoader:
    def __init__(self, meta_path, text_path, summary_path, corpus_filter):
        self.meta_path = meta_path
        self.text_path = text_path
        self.summary_path = summary_path
        self.corpus_filter = corpus_filter

        self.meta_corpus = pd.DataFrame()
        self.corpus = []
        self.tokenized_corpus = []
        self.vocab2idx = {}

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

        tokenized_text_corpus = []
        tokenized_summary_corpus = []

        vocab2idx = {}
        mini_corpus_cnt = 0 # TODO: Remove mini-corpus

        for id in self.meta_corpus["ID"]:
            if mini_corpus_cnt < 200: # TODO: Remove mini-corpus
                text_file = self.text_path + id + ".txt"
                summary_file = self.summary_path + id + ".txt"

                try:
                    text = self.read_text(text_file)
                    summary = self.read_text(summary_file)

                    tokenized_text = word_tokenize(text)
                    tokenized_summary = word_tokenize(summary)

                    if len(text) >= 1000:
                        for word in tokenized_text:
                            if word not in vocab2idx:
                                vocab2idx[word] = len(vocab2idx)

                        for word in tokenized_summary:
                            if word not in vocab2idx:
                                vocab2idx[word] = len(vocab2idx)

                        text_corpus.append(text)
                        summary_corpus.append(summary)

                        tokenized_text_corpus.append(tokenized_text)
                        tokenized_summary_corpus.append(tokenized_summary)

                        mini_corpus_cnt += 1 # TODO: Remove mini-corpus

                except Exception as e:
                    print(e)

        self.vocab2idx = vocab2idx
        self.corpus = list(zip(text_corpus, summary_corpus))
        self.tokenized_corpus = list(zip(tokenized_text_corpus, tokenized_summary_corpus))

    @staticmethod
    def read_text(file_path):
        with open(file_path, "r", encoding="utf8", errors="ignore") as f:
            lines = f.readlines()
            text = "\n".join(lines)
            f.close()

        return text
