# Imports
import gc
import csv
import torch
import psutil
import datasets
import pandas as pd
import transformers
import tf2tf_tud_gpu_config as config
import tf2tf_tud_gpu_helpers as helpers

from datasets import ClassLabel


# Variables
model = config.model
tokenizer = transformers.BertTokenizer.from_pretrained(model)
batch_size = 16


# Main
def main():
    text = config.text_english

    tf2tf = transformers.EncoderDecoderModel.from_pretrained(
        config.path_model
    )

    tf2tf = helpers.configure_model(tf2tf, tokenizer)
    tf2tf.to("cuda")

    rouge = datasets.load_metric("rouge")
    helpers.empty_cache()

    test_data = datasets.arrow_dataset.Dataset.from_pandas(
        pd.DataFrame([[text], [None]], columns=[
                     "article", "highlights"])  # TODO: Test
    )

    summary = test_data.map(
        generate_summary,
        batched=True,
        batch_size=batch_size
    )

    print(f"HYP: {summary[0]['pred_summary']}")


if __name__ == "__main__":
    main()
