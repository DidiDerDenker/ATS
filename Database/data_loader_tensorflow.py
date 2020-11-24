# Imports
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


# Methods
def iterate_dataset(ds, name_text, name_summary):
    cnt = 0

    for entry in tfds.as_numpy(ds):
        cnt += 1
        # text, summary = str(entry[name_text])[2:-1], str(entry[name_summary])[2:-1]
        # print(text)
        # print(summary)
        # exit()

        # TODO: Translate both
        # TODO: Generate id
        # TODO: Export id to text-files
        # TODO: Export id to summary-files
        # TODO: Export meta-file, include name "tensorflow"
        # TODO: Update nlp-pipeline, generate report

    print(cnt)


# Main
def main():
    ds_wikihow, info = tfds.load("wikihow", split="train", with_info=True)
    # iterate_dataset(ds_wikihow, "text", "headline")

    ds_gigaword, info = tfds.load("gigaword", split="train", with_info=True)
    # iterate_dataset(ds_gigaword, "document", "summary")

    ds_cnn_dailymail, info = tfds.load("cnn_dailymail", split="train", with_info=True)
    iterate_dataset(ds_cnn_dailymail, "article", "highlights")


if __name__ == "__main__":
    main()
