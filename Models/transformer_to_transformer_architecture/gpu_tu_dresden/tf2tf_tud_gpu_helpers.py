# Imports
import gc
import csv
import torch
import psutil
import datasets
import transformers
import pandas as pd
import tf2tf_tud_gpu_config as config

from datasets import ClassLabel


# Methods
def load_data(language, ratio_corpus_wiki=0.0, ratio_corpus_news=0.0, ratio_corpus_mlsum=0.0):
    if str(language) == "english":
        train_data = datasets.load_dataset(
            "cnn_dailymail", "3.0.0", split="train")
        val_data = datasets.load_dataset(
            "cnn_dailymail", "3.0.0", split="validation[:10%]")
        test_data = datasets.load_dataset(
            "cnn_dailymail", "3.0.0", split="test[:5%]")

        train_data = train_data.rename_column("article", "text")
        train_data = train_data.rename_column("highlights", "summary")
        val_data = val_data.rename_column("article", "text")
        val_data = val_data.rename_column("highlights", "summary")
        test_data = test_data.rename_column("article", "text")
        test_data = test_data.rename_column("highlights", "summary")

        return train_data, val_data, test_data

    elif str(language) == "german":
        data_txt, data_ref = [], []

        # CORPUS: WIKI
        with open("./data_train.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
            next(reader, None)

            for row in reader:
                data_txt.append(row[0])
                data_ref.append(row[1])

        ds_wiki = datasets.arrow_dataset.Dataset.from_pandas(
            pd.DataFrame(
                list(zip(data_txt, data_ref)),
                columns=["text", "summary"]
            )
        )

        # CORPUS: NEWS
        df_news = pd.read_excel("./data_train_test.xlsx", engine="openpyxl")
        df_news = df_news[["article", "highlights"]]
        df_news.columns = ["text", "summary"]
        df_news = df_news[~df_news["summary"].str.contains("ZEIT")]
        df_news = df_news.dropna()
        ds_news = datasets.arrow_dataset.Dataset.from_pandas(df_news)
        ds_news = ds_news.remove_columns("__index_level_0__")

        # CORPUS: MLSUM
        ds_mlsum = datasets.load_dataset("mlsum", "de", split="train")
        ds_mlsum = ds_mlsum.remove_columns(["topic", "url", "title", "date"])

        text_corpus_mlsum = []
        summary_corpus_mlsum = []

        for entry in ds_mlsum:
            text = entry["text"]
            summary = entry["summary"]

            if summary in text:
                text = text[len(summary) + 1:len(text)]

            text_corpus_mlsum.append(text)
            summary_corpus_mlsum.append(summary)

        ds_mlsum = datasets.arrow_dataset.Dataset.from_pandas(
            pd.DataFrame(
                list(zip(text_corpus_mlsum, summary_corpus_mlsum)),
                columns=["text", "summary"]
            )
        )

        # ACTION: CONCAT
        german_data = datasets.concatenate_datasets([
            ds_wiki.select(
                range(0, int(len(ds_wiki) * ratio_corpus_wiki))),
            ds_news.select(
                range(0, int(len(ds_news) * ratio_corpus_news))),
            ds_mlsum.select(
                range(0, int(len(ds_mlsum) * ratio_corpus_mlsum)))
        ])

        german_data = german_data.shuffle()

        # ACTION: SPLIT
        train_size = int(len(german_data) * 0.800)
        valid_size = int(len(german_data) * 0.100)
        test_size = int(len(german_data) * 0.100)

        train_data = german_data.select(
            range(0, train_size))
        val_data = german_data.select(
            range(train_size, train_size + valid_size))
        test_data = german_data.select(
            range(train_size + valid_size, train_size + valid_size + test_size))

        del german_data

        return train_data.shuffle(), val_data.shuffle(), test_data.shuffle()


def test_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    print("Device:", device)
    print("Version:", torch.__version__)


def explore_corpus(data):
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


def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    psutil.virtual_memory()

    # print(torch.cuda.get_device_properties(0).total_memory)
    # print(torch.cuda.memory_reserved(0))
    # print(torch.cuda.memory_allocated(0))


def load_tokenizer_and_model(from_checkpoint=False):
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
            tf2tf = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                config.model_name
            )

        else:
            tf2tf = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
                config.model_name, config.model_name, tie_encoder_decoder=True
            )

    return tokenizer, tf2tf


def configure_model(tf2tf, tokenizer):
    tf2tf.config.decoder_start_token_id = tokenizer.cls_token_id
    tf2tf.config.bos_token_id = tokenizer.bos_token_id
    tf2tf.config.eos_token_id = tokenizer.sep_token_id
    tf2tf.config.pad_token_id = tokenizer.pad_token_id
    # tf2tf.config.vocab_size = tf2tf.config.encoder.vocab_size

    tf2tf.config.max_length = 128
    tf2tf.config.min_length = 56
    tf2tf.config.no_repeat_ngram_size = 3
    tf2tf.config.early_stopping = True
    tf2tf.config.length_penalty = 2.0
    tf2tf.config.num_beams = 2

    return tf2tf
