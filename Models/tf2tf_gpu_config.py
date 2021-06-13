language: str = "english"  # english, german, multilingual
model_name: str = "bert-base-multilingual-cased"
tokenizer_name: str = "bert-base-multilingual-cased"
batch_size: int = 16

ratio_corpus_wik: float = 1.0
ratio_corpus_nws: float = 1.0
ratio_corpus_mls: float = 1.0
ratio_corpus_eng: float = 1.0

path_output: str = "/scratch/ws/1/davo557d-ws_project/"
path_checkpoint: str = "/scratch/ws/1/davo557d-ws_project/checkpoint-100000"

train_size: float = 0.900
val_size: float = 0.025
test_size: float = 0.075

'''
- bert-base-multilingual-cased
- deepset/gbert-base
- xlm-roberta-base
- facebook/mbart-large-cc25
'''
