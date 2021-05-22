# Imports
import datasets
import pandas as pd
import transformers
import tf2tf_tud_gpu_config as config
import tf2tf_tud_gpu_helpers as helpers


# Main
batch_size = config.batch_size
model = config.model
tokenizer = config.tokenizer

tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(
    tokenizer  # BertTokenizer
)

tf2tf = transformers.EncoderDecoderModel.from_pretrained(
    config.path_checkpoint
)

tf2tf.to("cuda")
text = None

if config.language == "english":
    text = config.text_english

elif config.language == "german":
    text = config.text_german

temp = {
    "article": [text, text],
    "hightlights": ["Zusammenfassung", "Zusammenfassung"]
}

helpers.test_cuda()
helpers.empty_cache()
rouge = datasets.load_metric("rouge")

test_data = datasets.arrow_dataset.Dataset.from_pandas(
    pd.DataFrame.from_dict(
        temp, columns=["article", "highlights"], orient="index"
    )
)


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
    batch_size=batch_size
)

print(f"HYP: {summary[0]['pred_summary']}")
