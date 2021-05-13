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


# Main
batch_size = 16
model = config.model
tokenizer = transformers.BertTokenizer.from_pretrained(model)

tf2tf = transformers.EncoderDecoderModel.from_pretrained(
    config.path_checkpoint
)

train_data, val_data, test_data = helpers.load_data(
    language=config.language,
    corpus_wiki=config.corpus_wiki,
    corpus_news=config.corpus_news
)

helpers.test_cuda()
helpers.explore_corpus(train_data)
helpers.empty_cache()
helpers.configure_model(tf2tf, tokenizer)
rouge = datasets.load_metric("rouge")


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


results = test_data.map(
    generate_summary,
    batched=True,
    batch_size=batch_size
)

'''
print(f"HYP: {results[0]['pred_summary']}")
print(f"REF: {results[0]['highlights']}")
'''

print(
    rouge.compute(
        predictions=results["pred_summary"],
        references=results["highlights"],
        rouge_types=["rouge2"]
    )["rouge2"].mid
)
