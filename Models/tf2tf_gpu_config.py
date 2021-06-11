language = "english"  # english, german, multilingual
model_name = "facebook/mbart-large-cc25"
tokenizer_name = "facebook/mbart-large-cc25"
batch_size = 8  # 16

ratio_corpus_wiki = 1.00
ratio_corpus_news = 1.00
ratio_corpus_mlsum = 1.00
ratio_corpus_eng = 0.10

path_output = "/scratch/ws/1/davo557d-ws_project/"
path_checkpoint = "/scratch/ws/1/davo557d-ws_project/checkpoint-100000"

text_english = "..."
text_german = "..."

'''
- bert-base-multilingual-cased
- deepset/gbert-base
- xlm-roberta-base
- facebook/mbart-large-cc25
'''
