# Imports
import datasets
import transformers
import tf2tf_tud_gpu_config as config
import tf2tf_tud_gpu_helpers as helpers


# Main
tokenizer, tf2tf = helpers.load_tokenizer_and_model(from_checkpoint=True)

train_data, val_data, test_data = helpers.load_data(
    language=config.language,
    ratio_corpus_wiki=config.ratio_corpus_wiki,
    ratio_corpus_news=config.ratio_corpus_news,
    ratio_corpus_mlsum=config.ratio_corpus_mlsum
)

helpers.test_cuda()
helpers.explore_corpus(train_data)
helpers.empty_cache()
rouge = datasets.load_metric("rouge")

tf2tf = helpers.configure_model(tf2tf, tokenizer)
tf2tf.to("cuda")


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
    batch_size=config.batch_size
)

print(
    rouge.compute(
        predictions=results["pred_summary"],
        references=results["highlights"],
        rouge_types=["rouge2"]
    )["rouge2"].mid
)
