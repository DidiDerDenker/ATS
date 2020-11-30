# Imports
import sys
import uuid
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from collections import Counter


# Global Variables
TEXT_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
SUMMARY_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\summary_files\\"
META_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"


# Methods
def export_meta_file(name, log):
    global META_PATH

    meta_path = META_PATH + name + ".xlsx"
    df = pd.DataFrame(data=log, columns=["ID", "Title"])
    df = df.sort_values(by=["Title"])
    df[["ID", "Title"]].to_excel(meta_path, index=False)
    print("Export done...")


def iterate_dataset(ds, name_text, name_summary, meta_name):
    global TEXT_PATH
    global SUMMARY_PATH
    global META_PATH

    log = []
    iteration_num = 0
    export_num = 1

    for entry in tfds.as_numpy(ds):
        text = str(entry[name_text], "utf-8")
        summary = str(entry[name_summary], "utf-8")

        id = str(uuid.uuid4()).upper()
        text_path = TEXT_PATH + id + ".txt"
        summary_path = SUMMARY_PATH + id + ".txt"

        try:
            with open(text_path, "w", encoding="utf8") as f:
                f.write(text)
                f.close()

            with open(summary_path, "w", encoding="utf8") as f:
                f.write(summary)
                f.close()

            log.append((id, "-"))
            iteration_num += 1

        except Exception as e:
            print(e)

        if iteration_num % 100000 == 0:
            name = meta_name + "_" + str(export_num)
            export_meta_file(name, log)
            export_num += 1
            log = []

    name = meta_name + "_" + str(export_num)
    export_meta_file(name, log)


# Main
def main():
    # ds_wikihow, info = tfds.load("wikihow", split="train", with_info=True)
    # iterate_dataset(ds_wikihow, "text", "headline", "data_tensorflow_wikihow")

    # ds_gigaword, info = tfds.load("gigaword", split="train", with_info=True)
    # iterate_dataset(ds_gigaword, "document", "summary", "data_tensorflow_gigaword")

    ds_cnn_dailymail, info = tfds.load("cnn_dailymail", split="train", with_info=True) # TODO: Clear and parse texts
    iterate_dataset(ds_cnn_dailymail, "article", "highlights", "data_tensorflow_cnn_dailymail")


if __name__ == "__main__":
    main()
