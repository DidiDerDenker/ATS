# Imports
import datasets
import transformers
import tf2tf_gpu_config as config
import tf2tf_gpu_helpers as helpers


# Main
rouge = datasets.load_metric("rouge")

if "mbart" in config.model_name:
    data, _, _ = helpers.load_data()
    model = transformers.pipeline("summarization")

    list_of_texts = list(data["text"])
    list_of_candidates = list(data["summary"])
    list_of_summaries, list_of_predictions = [], []

    for i in range(0, len(list_of_texts) - 1):
        try:
            summary = model(
                str(list_of_texts[i])[0:1024],
                max_length=156,
                min_length=64,
                do_sample=False
            )[0]["summary_text"]

            list_of_predictions.append(summary)
            list_of_summaries.append(list_of_candidates[i])

        except Exception as e:
            print(e)

    print(
        rouge.compute(
            predictions=list_of_predictions,
            references=list_of_summaries,
            rouge_types=["rouge2"]
        )["rouge2"].mid
    )

else:
    tokenizer, tf2tf = helpers.load_tokenizer_and_model(from_checkpoint=True)
    train_data, val_data, test_data = helpers.load_data()

    tf2tf = helpers.configure_model(tf2tf, tokenizer)
    tf2tf.to("cuda")

    def generate_summary(batch):
        inputs = tokenizer(
            batch["text"],
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
            references=results["summary"],
            rouge_types=["rouge2"]
        )["rouge2"].mid
    )
