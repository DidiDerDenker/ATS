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
def process_data_to_model_inputs(batch):
    encoder_max_length = 512
    decoder_max_length = 128

    inputs = tokenizer(batch["article"], padding="max_length",
                       truncation=True, max_length=encoder_max_length)

    outputs = tokenizer(batch["highlights"], padding="max_length",
                        truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]
                       for labels in batch["labels"]]

    return batch


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str,
        references=label_str,
        rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# Main
def main():
    train_data, val_data, test_data = helpers.load_data(
        language=config.language,
        corpus_wiki=config.corpus_wiki,
        corpus_news=config.corpus_news
    )

    helpers.test_cuda()
    helpers.explore_corpus(train_data)

    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["article", "highlights"]
    )

    train_data.set_format(
        type="torch",
        columns=["input_ids",
                 "attention_mask",
                 "decoder_input_ids",
                 "decoder_attention_mask",
                 "labels"]
    )

    val_data = val_data.map(
        process_data_to_model_inputs,
        batched=True,
        remove_columns=["article", "highlights"]
    )

    val_data.set_format(
        type="torch",
        columns=["input_ids",
                 "attention_mask",
                 "decoder_input_ids",
                 "decoder_attention_mask",
                 "labels"]
    )

    tf2tf = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
        model, model, tie_encoder_decoder=False
    )

    tf2tf = helpers.configure_model(tf2tf, tokenizer)
    rouge = datasets.load_metric("rouge")
    helpers.empty_cache()
    tf2tf.to("cuda")

    training_args = transformers.Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        output_dir=config.path_output,
        warmup_steps=1000,
        save_steps=2000,
        logging_steps=1000,
        eval_steps=2000,
        save_total_limit=1,
        fp16=True,
    )

    trainer = transformers.Seq2SeqTrainer(
        model=tf2tf,
        args=training_args,
        compute_metrics=helpers.compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == "__main__":
    main()