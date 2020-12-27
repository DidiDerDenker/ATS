# Imports
from data_loader import DataLoader
from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer
from rouge import Rouge


# Global Variables
META_PATH = "C:\\Temp\\Corpus\\meta_files\\"
TEXT_PATH = "C:\\Temp\\Corpus\\text_files\\"
SUMMARY_PATH = "C:\\Temp\\Corpus\\summary_files\\"


# Methods
def process_transformers(text_corpus):
    pretrained_model = "t5-base" # ["bert-base-cased", "bert-large-cased", "albert-large-v2"]
    model = AutoModelWithLMHead.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary_corpus = [summarizer(text, max_length=0.25 * len(text))[0]["summary_text"] for text in text_corpus]

    return summary_corpus


def print_rouge_scores(scores):
    rouge_1 = scores["rouge-1"]
    rouge_2 = scores["rouge-2"]
    rouge_l = scores["rouge-l"]

    print(f"Rouge-1:\tf: {rouge_1['f']:.4f} | p: {rouge_1['p']:.4f} | r: {rouge_1['r']:.4f}")
    print(f"Rouge-2:\tf: {rouge_2['f']:.4f} | p: {rouge_2['p']:.4f} | r: {rouge_2['r']:.4f}")
    print(f"Rouge-l:\tf: {rouge_l['f']:.4f} | p: {rouge_l['p']:.4f} | r: {rouge_l['r']:.4f}")


# Main
def main():
    global META_PATH
    global TEXT_PATH
    global SUMMARY_PATH

    ''' DATA-SECTION '''
    corpus_filter = ["wikihow"] # ["cnn_dailymail", "wikihow"]
    instance = DataLoader(META_PATH, TEXT_PATH, SUMMARY_PATH, corpus_filter)
    X_train, y_train, X_test, y_test = instance.train_test_split(size=0.75, shuffle=True)

    ''' MODEL-SECTION '''
    y_hyps = process_transformers(X_test)
    y_refs = y_test

    ''' SCORE-SECTION '''
    instance = Rouge()
    scores = instance.get_scores(y_hyps, y_refs, avg=True)
    print_rouge_scores(scores)


if __name__ == "__main__":
    main()
