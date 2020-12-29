# Imports
import random
import models.seq_to_seq_attention.preprocessing as pp_ssa
# import models.bert_encoder_transformer_decoder.preprocessing as pp_betd
# import models.rl_seq_to_seq.preprocessing as pp_rlss

from data_loader import DataLoader
from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer
from rouge import Rouge


# Global Variables
META_PATH = "C:\\Temp\\Corpus\\meta_files\\"
TEXT_PATH = "C:\\Temp\\Corpus\\text_files\\"
SUMMARY_PATH = "C:\\Temp\\Corpus\\summary_files\\"
GLOVE_PATH = "C:\\Users\\didid\\GitHub-Respository\\AutomaticTextSummarization\\Models\\models\\seq_to_seq_attention\\embeddings\\glove.6B.100d.txt"
JSON_PATH = "C:\\Users\\didid\\GitHub-Respository\\AutomaticTextSummarization\\Models\\models\\seq_to_seq_attention\\data\\processed_data.json"
OPTION = 2


# Methods
def preprocess_transformers(corpus, size, shuffle):
    train_data = []
    test_data = []

    if shuffle:
        random.shuffle(corpus)

    for pair in corpus:
        selector = random.random()
        train_data.append(pair) if selector < size else test_data.append(pair)

    X_train = [pair[0] for pair in train_data]
    y_train = [pair[1] for pair in train_data]

    X_test = [pair[0] for pair in test_data]
    y_test = [pair[1] for pair in test_data]

    return X_train, y_train, X_test, y_test


def process_transformers(text_corpus):
    pretrained_model = "t5-base" # ["bert-base-cased", "bert-large-cased", "albert-large-v2"]
    model = AutoModelWithLMHead.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary_corpus = [summarizer(text, max_length=int(0.15 * len(text)))[0]["summary_text"] for text in text_corpus]

    return summary_corpus


# https://nlp.stanford.edu/projects/glove/
def preprocess_seq_to_seq_attention(corpus, vocab2idx):
    global GLOVE_PATH
    global JSON_PATH

    embeddings = pp_ssa.load_embeddings(GLOVE_PATH)
    embeddings, vocab2idx = pp_ssa.setup_vocabulary(embeddings, vocab2idx)
    vec_texts, vec_summaries = pp_ssa.vectorize_and_shuffle_data(corpus, vocab2idx)
    pp_ssa.prepare_batches(vec_texts, vec_summaries, embeddings, vocab2idx, JSON_PATH)
    exit()


# https://github.com/JRC1995/Abstractive-Summarization
def process_seq_to_seq_attention(text_corpus):
    global JSON_PATH

    # TODO: Next chapter

    return summary_corpus


def preprocess_bert_encoder_transformer_decoder():
    exit()


# https://github.com/santhoshkolloju/Abstractive-Summarization-With-Transfer-Learning
def process_bert_encoder_transformer_decoder(text_corpus):
    model = None
    summary_corpus = None

    return summary_corpus


def preprocess_rl_seq_to_seq():
    exit()


# https://github.com/yaserkl/RLSeq2Seq, https://arxiv.org/abs/1805.09461
def process_rl_seq_to_seq(text_corpus):
    model = None
    summary_corpus = None

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
    global OPTION


    ''' TRANSFORMERS '''
    if OPTION == 1:
        corpus_filter = ["cnn_dailymail", "wikihow"]
        instance = DataLoader(META_PATH, TEXT_PATH, SUMMARY_PATH, corpus_filter)

        X_train, y_train, X_test, y_test = preprocess_transformers(instance.corpus, size=0.75, shuffle=True)
        y_hyps = process_transformers(X_test)
        y_refs = y_test

        instance = Rouge()
        scores = instance.get_scores(y_hyps, y_refs, avg=True)
        print_rouge_scores(scores)


    ''' SEQ-TO-SEQ-ATTENTION '''
    if OPTION == 2:
        corpus_filter = ["wikihow"] # ["cnn_dailymail", "wikihow"]
        instance = DataLoader(META_PATH, TEXT_PATH, SUMMARY_PATH, corpus_filter)

        preprocess_seq_to_seq_attention(instance.tokenized_corpus, instance.vocab2idx)
        process_seq_to_seq_attention() # TODO: Train model
        # scores... # TODO: Evaluate model


    ''' BERT-ENCODER-TRANSFORMER-DECODER '''
    if OPTION == 3:
        corpus_filter = ["cnn_dailymail", "wikihow"]
        instance = DataLoader(META_PATH, TEXT_PATH, SUMMARY_PATH, corpus_filter)

        preprocess_bert_encoder_transformer_decoder()
        process_bert_encoder_transformer_decoder() # TODO: Train model
        # scores... # TODO: Evaluate model


    ''' RL-SEQ-TO-SEQ '''
    if OPTION == 4:
        corpus_filter = ["cnn_dailymail", "wikihow"]
        instance = DataLoader(META_PATH, TEXT_PATH, SUMMARY_PATH, corpus_filter)

        preprocess_rl_seq_to_seq()
        process_rl_seq_to_seq() # TODO: Train model
        # scores... # TODO: Evaluate model


if __name__ == "__main__":
    main()
