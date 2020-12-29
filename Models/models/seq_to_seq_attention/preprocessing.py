# Imports
import numpy as np
import random
import json


# Methods
def load_embeddings(file_path):
    vocab2emb = {}

    with open(file_path) as f:
        for line in f:
            row = line.strip().split(" ")
            word = row[0].lower()

            if word not in vocab2emb:
                vocab2emb[word] = np.asarray(row[1:], np.float32)

    return vocab2emb


def setup_vocabulary(vocab2emb, vocab2idx):
    vocab = []
    embeddings = []
    special_tags = ["<UNK>", "<PAD>", "<EOS>"]

    for word in vocab2idx:
        if word in vocab2emb:
            vocab.append(word)
            embeddings.append(vocab2emb[word])

    for special_tag in special_tags:
        vocab.append(special_tag)
        embeddings.append(np.random.rand(len(embeddings[0]), ))

    embeddings = np.asarray(embeddings, np.float32)
    vocab2idx = {word: idx for idx, word in enumerate(vocab)}

    return embeddings, vocab2idx


def vectorize_and_shuffle_data(corpus, vocab2idx):
    vec_texts = []
    vec_summaries = []

    texts = [pair[0] for pair in corpus]
    summaries = [pair[1] for pair in corpus]

    for text, summary in zip(texts, summaries):
        vec_texts.append([vocab2idx.get(word, vocab2idx["<UNK>"]) for word in text])
        vec_summaries.append([vocab2idx.get(word, vocab2idx["<UNK>"]) for word in summary])

    random.seed(101)

    texts_idx = [idx for idx in range(len(vec_texts))]
    random.shuffle(texts_idx)

    vec_texts = [vec_texts[idx] for idx in texts_idx]
    vec_summaries = [vec_summaries[idx] for idx in texts_idx]

    return vec_texts, vec_summaries


def prepare_batches(vec_texts, vec_summaries, embeddings, vocab2idx, output_path):
    X_test = vec_texts[0:10000]
    y_test = vec_summaries[0:10000]

    X_val = vec_texts[10000:20000]
    y_val = vec_summaries[10000:20000]

    X_train = vec_texts[20000:]
    y_train = vec_summaries[20000:]

    train_batches_text, train_batches_summary, \
    train_batches_true_text_len, train_batches_true_summary_len \
        = bucket_and_batch(X_train, y_train, vocab2idx)

    val_batches_text, val_batches_summary, \
    val_batches_true_text_len, val_batches_true_summary_len \
        = bucket_and_batch(X_val, y_val, vocab2idx)

    test_batches_text, test_batches_summary, \
    test_batches_true_text_len, test_batches_true_summary_len \
        = bucket_and_batch(X_test, y_test, vocab2idx)

    d = {}

    d["vocab"] = vocab2idx
    d["embd"] = embeddings.tolist()
    d["train_batches_text"] = train_batches_text
    d["test_batches_text"] = test_batches_text
    d["val_batches_text"] = val_batches_text
    d["train_batches_summary"] = train_batches_summary
    d["test_batches_summary"] = test_batches_summary
    d["val_batches_summary"] = val_batches_summary
    d["train_batches_true_text_len"] = train_batches_true_text_len
    d["val_batches_true_text_len"] = val_batches_true_text_len
    d["test_batches_true_text_len"] = test_batches_true_text_len
    d["train_batches_true_summary_len"] = train_batches_true_summary_len
    d["val_batches_true_summary_len"] = val_batches_true_summary_len
    d["test_batches_true_summary_len"] = test_batches_true_summary_len

    with open(output_path, "w") as f:
        json.dump(d, f)


def bucket_and_batch(texts, summaries, vocab2idx, batch_size=32):
    text_lens = [len(text) for text in texts]
    sortedidx = np.flip(np.argsort(text_lens), axis=0)
    texts = [texts[idx] for idx in sortedidx]
    summaries = [summaries[idx] for idx in sortedidx]

    batches_text = []
    batches_summary = []
    batches_true_text_len = []
    batches_true_summary_len = []

    i = 0

    while i < (len(texts) - batch_size):
        max_len = len(texts[i])

        batch_text = []
        batch_summary = []
        batch_true_text_len = []
        batch_true_summary_len = []

        for j in range(batch_size):
            padded_text = texts[i + j]
            padded_summary = summaries[i + j]

            batch_true_text_len.append(len(texts[i + j]))
            batch_true_summary_len.append(len(summaries[i + j]) + 1)

            while len(padded_text) < max_len:
                padded_text.append(vocab2idx["<PAD>"])

            padded_summary.append(vocab2idx["<EOS>"])
            summary_max_len = 30 # TODO: Calculate summary_max_len dynamically

            while len(padded_summary) < summary_max_len + 1:
                padded_summary.append(vocab2idx["<PAD>"])

            batch_text.append(padded_text)
            batch_summary.append(padded_summary)

        batches_text.append(batch_text)
        batches_summary.append(batch_summary)
        batches_true_text_len.append(batch_true_text_len)
        batches_true_summary_len.append(batch_true_summary_len)

        i += batch_size

    return batches_text, batches_summary, batches_true_text_len, batches_true_summary_len
