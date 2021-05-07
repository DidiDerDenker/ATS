# Imports
import psutil
import gc
import pandas as pd
import datasets
import transformers
import csv
import torch

from datasets import ClassLabel


# Test CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print("Device:", device)
print("Version:", torch.__version__)


# Load swiss data
data_txt = []
data_ref = []

with open("./data_train.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
    next(reader, None)

    for row in reader:
        data_txt.append(row[0])
        data_ref.append(row[1])

tuples = list(zip(data_txt, data_ref))
dataframe = pd.DataFrame(tuples, columns=["article", "highlights"])


# Load german data
tuples = pd.read_excel("./data_train_test.xlsx", engine="openpyxl")
del tuples["Unnamed: 0"]
dataframe = pd.concat([dataframe, tuples])
dataframe = dataframe.dropna()


# Clean redactional data
print(len(dataframe))
dataframe = dataframe[~dataframe["highlights"].str.contains("ZEIT")]
print(len(dataframe))


# Concat swiss and german data
german_data = datasets.arrow_dataset.Dataset.from_pandas(
    dataframe[["article", "highlights"]]
)
german_data = german_data.shuffle()
print(dataframe.head(10))


# Split data
train_size = int(len(dataframe) * 0.8)
valid_size = int(len(dataframe) * 0.1)
test_size = int(len(dataframe) * 0.1)

train_data = german_data.select(range(0, train_size))
val_data = german_data.select(range(train_size, train_size + valid_size))
test_data = german_data.select(range(train_size + valid_size, len(dataframe)))


# Load english data
'''
train_data = datasets.load_dataset(
    "cnn_dailymail", "3.0.0", split="train")
val_data = datasets.load_dataset(
    "cnn_dailymail", "3.0.0", split="validation[:10%]")
test_data = datasets.load_dataset(
    "cnn_dailymail", "3.0.0", split="test[:5%]")
'''


# Explore corpus
df = pd.DataFrame(train_data)

text_list = []
summary_list = []

for index, row in df.iterrows():
    text = row["article"]
    summary = row["highlights"]
    text_list.append(len(text))
    summary_list.append(len(summary))

print(sum(text_list) / len(text_list))
print(sum(summary_list) / len(summary_list))


# Explore corpus
train_data.info.description
df = pd.DataFrame(train_data[:1])

for column, typ in train_data.features.items():
    if isinstance(typ, ClassLabel):
        df[column] = df[column].transform(lambda i: typ.names[i])


# Load tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)
print(type(tokenizer))


# Prepare data
encoder_max_length = 512
decoder_max_length = 128
batch_size = 16


def process_data_to_model_inputs(batch):
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


# Training data
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


# Validation data
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


# Load models
tf2tf = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-multilingual-cased",
    "bert-base-multilingual-cased",
    tie_encoder_decoder=False
)


# Configure models
tf2tf.config.decoder_start_token_id = tokenizer.cls_token_id
tf2tf.config.bos_token_id = tokenizer.bos_token_id
tf2tf.config.eos_token_id = tokenizer.sep_token_id
tf2tf.config.pad_token_id = tokenizer.pad_token_id
tf2tf.config.vocab_size = tf2tf.config.encoder.vocab_size

print(tf2tf.config.decoder_start_token_id)
print(tf2tf.config.eos_token_id)
print(tf2tf.config.vocab_size)


# Configure beam search
tf2tf.config.max_length = 142
tf2tf.config.min_length = 56
tf2tf.config.no_repeat_ngram_size = 3
tf2tf.config.early_stopping = True
tf2tf.config.length_penalty = 2.0
tf2tf.config.num_beams = 4


# Prepare metric
rouge = datasets.load_metric("rouge")


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


# Empty cache
gc.collect()
torch.cuda.empty_cache()
psutil.virtual_memory()


# Setup arguments
path_output = "/scratch/ws/1/davo557d-ws_project/"
tf2tf.to("cuda")

training_args = transformers.Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir=path_output,
    warmup_steps=1000,
    save_steps=2000,
    logging_steps=1000,
    eval_steps=2000,
    save_total_limit=1,
    fp16=True,
)


# Start training
trainer = transformers.Seq2SeqTrainer(
    model=tf2tf,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer
)

trainer.train()
