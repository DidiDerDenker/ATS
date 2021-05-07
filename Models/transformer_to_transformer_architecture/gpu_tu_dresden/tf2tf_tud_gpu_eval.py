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


# Load tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased")
print(type(tokenizer))


# Prepare data
encoder_max_length = 512
decoder_max_length = 128
batch_size = 16


# Load models
path_checkpoint = "/scratch/ws/1/davo557d-ws_project/checkpoint-52000"  # TODO: Set
tf2tf = transformers.EncoderDecoderModel.from_pretrained(path_checkpoint)
tf2tf.to("cuda")


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
        predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# Empty cache
gc.collect()
torch.cuda.empty_cache()
psutil.virtual_memory()


# Evaluate training
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

print(rouge.compute(predictions=results["pred_summary"],
                    references=results["highlights"],
                    rouge_types=["rouge2"])["rouge2"].mid)
