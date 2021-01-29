# Imports
import os
import re
import collections
import pickle
import numpy as np

from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


# Global Variables
path = "C:\\Temp\\MiniCorpus\\"
text_path = path + "text_files\\"
summary_path = path + "summary_files\\"

text_files = os.listdir(text_path)
summary_files = os.listdir(summary_path)
size = int(len(text_files) * 0.75)

train_text_files = text_files[0:size]
test_text_files = text_files[size:len(text_files)]
train_summary_files = summary_files[0:size]
test_summary_files = summary_files[size:len(summary_files)]


# Methods
def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)

    return sentence


def read_files(path, files):
    corpus = []

    for file in files:
        if file.endswith(".txt"):
            file_path = path + file

            with open(file_path, "r", encoding="utf8") as f:
                corpus.append(" ".join(f.readlines()))

    return corpus


def build_dict(step, toy=False):
    if step == "train":
        words = list()

        for sentence in read_files(text_path, train_text_files) + read_files(summary_path, train_summary_files):
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3

        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
    article_max_len = 1000 # TODO: Set
    summary_max_len = 200 # TODO: Set

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        text_list = read_files(text_path, train_text_files)
        summary_list = read_files(summary_path, train_summary_files)

    elif step == "valid":
        text_list = read_files(text_path, test_text_files)

    else:
        raise NotImplementedError

    x = [word_tokenize(d) for d in text_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x

    else:        
        y = [word_tokenize(d) for d in summary_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]

        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1

    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))

            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
    word_vec_list = list()

    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)

        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)
