# Imports
import csv
import datasets
import gc
import psutil
import pandas as pd
import torch
import transformers
import tf2tf_gpu_config as config

from datasets import ClassLabel
from typing import Tuple


# Methods
def load_data() -> Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    if config.language == "english":
        return load_english_data()

    if config.language == "german":
        return load_german_data()

    if config.language == "multilingual":
        return load_multilingual_data()


def load_english_data() -> Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    train_data = datasets.load_dataset(
        "cnn_dailymail", "3.0.0",
        split="train",
        ignore_verifications=True
    )

    val_data = datasets.load_dataset(
        "cnn_dailymail", "3.0.0",
        split="validation[:50%]",
        ignore_verifications=True
    )

    test_data = datasets.load_dataset(
        "cnn_dailymail", "3.0.0",
        split="test[:50%]",
        ignore_verifications=True
    )

    train_data = train_data.select(
        range(0, int(len(train_data) * config.ratio_corpus_eng))
    )

    train_data = train_data.rename_column("article", "text")
    train_data = train_data.rename_column("highlights", "summary")
    train_data = train_data.remove_columns("id")

    val_data = val_data.rename_column("article", "text")
    val_data = val_data.rename_column("highlights", "summary")
    val_data = val_data.remove_columns("id")

    test_data = test_data.rename_column("article", "text")
    test_data = test_data.rename_column("highlights", "summary")
    test_data = test_data.remove_columns("id")

    return train_data.shuffle(), val_data.shuffle(), test_data.shuffle()


def load_german_data() -> Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    ds_wik = load_corpus_wik()
    ds_nws = load_corpus_nws()
    ds_mls = load_corpus_mls()

    german_data = datasets.concatenate_datasets([
        ds_wik.select(
            range(0, int(len(ds_wik) * config.ratio_corpus_wik))),
        ds_nws.select(
            range(0, int(len(ds_nws) * config.ratio_corpus_nws))),
        ds_mls.select(
            range(0, int(len(ds_mls) * config.ratio_corpus_mls)))
    ])

    train_size = int(len(german_data) * config.train_size)
    valid_size = int(len(german_data) * config.val_size)
    test_size = int(len(german_data) * config.test_size)

    train_data = german_data.select(
        range(0, train_size)
    )

    val_data = german_data.select(
        range(train_size, train_size + valid_size)
    )

    test_data = german_data.select(
        range(train_size + valid_size, train_size + valid_size + test_size)
    )

    return train_data, val_data, test_data


def load_corpus_wik() -> datasets.Dataset:
    data_txt, data_ref = [], []

    with open("./data_train.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
        next(reader, None)

        for row in reader:
            data_txt.append(row[0])
            data_ref.append(row[1])

    df_wik = pd.DataFrame(
        list(zip(data_txt, data_ref)),
        columns=["text", "summary"]
    )

    ds_wik = datasets.arrow_dataset.Dataset.from_pandas(df_wik)

    return ds_wik.shuffle()


def load_corpus_nws() -> datasets.Dataset:
    df_nws = pd.read_excel("./data_train_test.xlsx", engine="openpyxl")
    df_nws = df_nws[["article", "highlights"]]
    df_nws.columns = ["text", "summary"]
    df_nws = df_nws[~df_nws["summary"].str.contains("ZEIT")]
    df_nws = df_nws.dropna()
    ds_nws = datasets.arrow_dataset.Dataset.from_pandas(df_nws)
    ds_nws = ds_nws.remove_columns("__index_level_0__")

    return ds_nws.shuffle()


def load_corpus_mls() -> datasets.Dataset:
    ds_mls = datasets.load_dataset("mlsum", "de", split="train")
    ds_mls = ds_mls.remove_columns(["topic", "url", "title", "date"])

    text_corpus_mls = []
    summary_corpus_mls = []

    for entry in ds_mls:
        text = entry["text"]
        summary = entry["summary"]

        if summary in text:
            text = text[len(summary) + 1:len(text)]

        text_corpus_mls.append(text)
        summary_corpus_mls.append(summary)

    df_mls = pd.DataFrame(
        list(zip(text_corpus_mls, summary_corpus_mls)),
        columns=["text", "summary"]
    )

    ds_mls = datasets.arrow_dataset.Dataset.from_pandas(df_mls)

    return ds_mls.shuffle()


def load_multilingual_data() -> Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    english_data, _, _ = load_english_data()
    german_data, _, _ = load_german_data()

    multilingual_data = datasets.concatenate_datasets([
        german_data, english_data
    ]).shuffle()

    train_size = int(len(multilingual_data) * config.train_size)
    valid_size = int(len(multilingual_data) * config.val_size)
    test_size = int(len(multilingual_data) * config.test_size)

    train_data = multilingual_data.select(
        range(0, train_size)
    )

    val_data = multilingual_data.select(
        range(train_size, train_size + valid_size)
    )

    test_data = multilingual_data.select(
        range(train_size + valid_size, train_size + valid_size + test_size)
    )

    return train_data, val_data, test_data


def test_cuda() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    print("Device:", device)
    print("Version:", torch.__version__)


def explore_corpus(data: datasets.Dataset) -> None:
    df = pd.DataFrame(data)

    text_list = []
    summary_list = []

    for index, row in df.iterrows():
        text = row["text"]
        summary = row["summary"]
        text_list.append(len(text))
        summary_list.append(len(summary))

    df = pd.DataFrame(data[:1])

    for column, typ in data.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])


def empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    psutil.virtual_memory()


def load_tokenizer_and_model(from_checkpoint: bool = False) -> Tuple[transformers.AutoTokenizer, transformers.EncoderDecoderModel]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer_name, strip_accent=False  # add_prefix_space=True
    )

    if from_checkpoint:
        if "mbart" in config.model_name:
            tf2tf = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                config.path_checkpoint
            )

        else:
            tf2tf = transformers.EncoderDecoderModel.from_pretrained(
                config.path_checkpoint
            )

    else:
        if "mbart" in config.model_name:
            tf2tf = transformers.MBartForConditionalGeneration.from_pretrained(
                config.model_name
            )

        else:
            tf2tf = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
                config.model_name, config.model_name, tie_encoder_decoder=True
            )

    return tokenizer, tf2tf


def configure_model(tf2tf: transformers.EncoderDecoderModel, tokenizer: transformers.AutoTokenizer):
    tf2tf.config.decoder_start_token_id = tokenizer.cls_token_id
    tf2tf.config.bos_token_id = tokenizer.bos_token_id
    tf2tf.config.eos_token_id = tokenizer.sep_token_id
    tf2tf.config.pad_token_id = tokenizer.pad_token_id

    tf2tf.config.max_length = 128
    tf2tf.config.min_length = 56
    tf2tf.config.no_repeat_ngram_size = 3
    tf2tf.config.early_stopping = True
    tf2tf.config.length_penalty = 2.0
    tf2tf.config.num_beams = 2

    return tf2tf
