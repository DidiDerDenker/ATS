# Imports
import sys
import uuid
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


# Global Variables
TEXT_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
SUMMARY_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\summary_files\\"
META_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"
EXPORT_PROGRESS = 0


# Methods
def iterate_dataset(ds, name_text, name_summary, meta_name):
    global TEXT_PATH
    global SUMMARY_PATH
    global META_PATH
    global EXPORT_PROGRESS

    log = []

    for entry in tfds.as_numpy(ds):
        text, summary = str(entry[name_text])[2:-1], str(entry[name_summary])[2:-1] # TODO: Translate
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
            sys.stdout.write("\rFile %i..." % (EXPORT_PROGRESS + 1))
            sys.stdout.flush()
            EXPORT_PROGRESS += 1

        except Exception as e:
            print(e)

    meta_path = META_PATH + meta_name + ".xlsx"
    df = pd.DataFrame(data=log, columns=["ID", "Title"])
    df = df.sort_values(by=["Title"])
    df[["ID", "Title"]].to_excel(meta_path, index=False)


# Main
def main():
    ds_wikihow, info = tfds.load("wikihow", split="train", with_info=True)
    iterate_dataset(ds_wikihow, "text", "headline", "data_tensorflow_wikihow")

    ds_gigaword, info = tfds.load("gigaword", split="train", with_info=True)
    iterate_dataset(ds_gigaword, "document", "summary", "data_tensorflow_gigaword")

    ds_cnn_dailymail, info = tfds.load("cnn_dailymail", split="train", with_info=True) # TODO: Clear and parse texts
    iterate_dataset(ds_cnn_dailymail, "article", "highlights", "data_tensorflow_cnn_dailymail")


if __name__ == "__main__":
    main()
