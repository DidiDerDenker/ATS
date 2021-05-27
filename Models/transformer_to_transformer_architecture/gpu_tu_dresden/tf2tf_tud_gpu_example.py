# Imports
import datasets
import pandas as pd
import transformers
import tf2tf_tud_gpu_config as config
import tf2tf_tud_gpu_helpers as helpers


# Main
tokenizer, tf2tf = helpers.load_tokenizer_and_model(from_checkpoint=True)

tf2tf = helpers.configure_model(tf2tf, tokenizer)
tf2tf.to("cuda")

text = None
parts = []


def split_long_texts(text):
    limit = 512

    if len(text) > limit:
        end_index = max([
            text.rfind(".", 0, limit),
            text.rfind("!", 0, limit),
            text.rfind("?", 0, limit)
        ])

        parts.append(text[0:end_index + 1].strip())
        text = text[end_index + 1:len(text)].strip()
        split_long_texts(text)

    else:
        parts.append(text)


text = config.text_english if config.language == "english" else config.text_german
split_long_texts(text)

if len(parts) > 1:
    article = parts
    highlights = [None] * len(parts)

else:
    parts = [text]
    article = [text] * 2
    highlights = [None] * 2

helpers.test_cuda()
helpers.empty_cache()
rouge = datasets.load_metric("rouge")

df = pd.DataFrame({"article": article, "highlights": highlights})
test_data = datasets.arrow_dataset.Dataset.from_pandas(df)


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


summary = test_data.map(
    generate_summary,
    batched=True,
    batch_size=config.batch_size
)

result = ""

for i in range(0, len(parts)):
    result = result + " " + summary[i]["pred_summary"]

print(result.strip())
