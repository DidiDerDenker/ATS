# Imports
import sys
import subprocess
import os
import random

from data_loader import DataLoader
from transformers import pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer
from rouge import Rouge


# Global Variables
META_PATH = "C:\\Temp\\Corpus\\meta_files\\"
TEXT_PATH = "C:\\Temp\\Corpus\\text_files\\"
SUMMARY_PATH = "C:\\Temp\\Corpus\\summary_files\\"


'''
Note: Commented snippets are scripts that have to be executed only once
Note: Methods of models can be run to evaluate the model, but not everyone is working right now
'''


# Methods
def process_transformers(corpus, shuffle=True, size=0.75):
    if shuffle:
        random.shuffle(corpus)

    X = [pair[0] for pair in corpus]
    y = [pair[1] for pair in corpus]

    '''
    train_data = []
    test_data = []

    for pair in text_corpus:
        selector = random.random()
        train_data.append(pair) if selector < size else test_data.append(pair)

    X_train = [pair[0] for pair in train_data]
    y_train = [pair[1] for pair in train_data]

    X_test = [pair[0] for pair in test_data]
    y_test = [pair[1] for pair in test_data]
    '''

    pretrained_model = "t5-base" # ["bert-base-cased", "bert-large-cased", "albert-large-v2"]
    model = AutoModelWithLMHead.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    y_hyps = []
    y_refs = y
    i = 1

    for text in X:
        y_hyps.append(summarizer(text, max_length=int(len(text) * 0.15))[0]["summary_text"])
        sys.stdout.write("\rSummarization %i..." % i)
        sys.stdout.flush()
        i += 1

    instance = Rouge()
    scores = instance.get_scores(y_hyps, y_refs, avg=True)
    print_rouge_scores(scores)


def process_seq_to_seq_with_attention(): # corpus, vocab2idx
    import models.seq_to_seq_with_attention.helpers as temp

    glove_file_path = "C:\\Users\\didid\\GitHub-Respository\\AutomaticTextSummarization\\Models\\models\\seq_to_seq_attention\\embeddings\\glove.6B.100d.txt"
    json_file_path = "C:\\Users\\didid\\GitHub-Respository\\AutomaticTextSummarization\\Models\\models\\seq_to_seq_attention\\data\\processed_data.json"

    '''
    embeddings = temp.load_embeddings(glove_file_path)
    embeddings, vocab2idx = temp.setup_vocabulary(embeddings, vocab2idx)
    vec_texts, vec_summaries = temp.vectorize_and_shuffle_data(corpus, vocab2idx)
    temp.prepare_batches(vec_texts, vec_summaries, embeddings, vocab2idx, json_file_path)
    '''

    y_hyps, y_refs = temp.model_notebook(json_file_path)

    instance = Rouge()
    scores = instance.get_scores(y_hyps, y_refs, avg=True)
    print_rouge_scores(scores)

    # TODO: Train the model properly

    exit()


def process_seq_to_seq_with_attention_library():
    cwd = os.path.dirname(os.path.realpath(__file__)) + "\\models\\seq_to_seq_with_attention_library"
    # subprocess.call("python train.py --glove --num_epochs 30", shell=True, cwd=cwd)
    # subprocess.call("python test.py", shell=True, cwd=cwd)

    # TODO: Train and test model
    # TODO: Load checkpoints and continue training
    # TODO: Evaluate scores, change export to result.txt to a text-wise export

    exit()


def process_bert_encoder_transformer_decoder():
    base_path = "C:\\Users\\didid\\GitHub-Respository\\AutomaticTextSummarization\\Models"
    cwd = base_path + "\\models\\bert_encoder_transformer_decoder\\src\\"
    log_path = base_path + "\\models\\bert_encoder_transformer_decoder\\logs\\"
    bert_path = base_path + "\\models\\bert_encoder_transformer_decoder\\bert_data\\"
    model_path = base_path + "\\models\\bert_encoder_transformer_decoder\\models\\"

    subprocess.call(
        "python train.py "
        "-mode train "
        "-accum_count 5 "
        "-batch_size 300 "
        "-bert_data_path " + str(bert_path) + " "
        "-dec_dropout 0.1 "
        "-log_file " + str(log_path) + "cnn_baseline.log "
        "-lr 0.05 "
        "-model_path " + str(model_path) + " "
        "-save_checkpoint_steps 2000 "
        "-seed 777 "
        "-sep_optim false "
        "-train_steps 200000 "
        "-use_bert_emb true "
        "-use_interval true "
        "-warmup_steps 8000 "
        "-visible_gpus 0,1,2,3 "
        "-max_pos 512 "
        "-report_every 50 "
        "-enc_hidden_size 512 "
        "-enc_layers 6 "
        "-enc_ff_size 2048 "
        "-enc_dropout 0.1 "
        "-dec_layers 6 "
        "-dec_hidden_size 512 "
        "-dec_ff_size 2048 "
        "-encoder baseline "
        "-task abs",
        shell=True, cwd=cwd
    )

    exit()


def process_seq_to_seq_with_rnn(text_corpus):
    exit()


def process_deep_reinforced_model_with_pytorch(text_corpus):
    exit()


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

    # corpus_filter = ["cnn_dailymail", "wikihow"]
    # instance = DataLoader(META_PATH, TEXT_PATH, SUMMARY_PATH, corpus_filter)

    ''' TRANSFORMERS '''
    # process_transformers(instance.corpus)

    ''' SEQ-TO-SEQ WITH ATTENTION '''
    # process_seq_to_seq_with_attention(instance.tokenized_corpus, instance.vocab2idx)

    ''' SEQ-TO-SEQ WITH ATTENTION LIBRARY '''
    # process_seq_to_seq_with_attention_library()

    ''' BERT-ENCODER-TRANSFORMER-DECODER '''
    process_bert_encoder_transformer_decoder()

    ''' DEEP-REINFORCED-MODEL WITH PYTORCH '''
    # process_deep_reinforced_model_with_pytorch(instance.corpus)


if __name__ == "__main__":
    main()
