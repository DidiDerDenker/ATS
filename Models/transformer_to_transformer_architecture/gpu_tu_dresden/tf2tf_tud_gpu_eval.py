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


# Methods
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


def generate_summary(batch):
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = tf2tf.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["pred_summary"] = output_str

    return batch


# Main
def main():
    train_data, val_data, test_data = helpers.load_data(
        language=config.language,
        corpus_wiki=config.corpus_wiki,
        corpus_news=config.corpus_news
    )

    helpers.test_cuda()
    helpers.explore_corpus(train_data)

    tf2tf = transformers.EncoderDecoderModel.from_pretrained(
        config.path_checkpoint
    )

    tf2tf = helpers.configure_model(tf2tf, tokenizer)
    tf2tf.to("cuda")

    rouge = datasets.load_metric("rouge")
    helpers.empty_cache()

    results = test_data.map(
        generate_summary,
        batched=True,
        batch_size=batch_size
    )

    print(f"HYP: {results[0]['pred_summary']}")
    print(f"REF: {results[0]['highlights']}")

    print(rouge.compute(predictions=results["pred_summary"],
                        references=results["highlights"],
                        rouge_types=["rouge2"])["rouge2"].mid)


if __name__ == "__main__":
    main()
