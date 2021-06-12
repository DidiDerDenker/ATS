language: str = "multilingual"  # english, german, multilingual
model_name: str = "xlm-roberta-base"
tokenizer_name: str = "xlm-roberta-base"
batch_size: int = 8  # 16

ratio_corpus_wik: float = 1.0
ratio_corpus_nws: float = 1.0
ratio_corpus_mls: float = 1.0
ratio_corpus_eng: float = 1.0

path_output: str = "/scratch/ws/1/davo557d-ws_project/"
path_checkpoint: str = "/scratch/ws/1/davo557d-ws_project/checkpoint-40000"

'''
- bert-base-multilingual-cased
- deepset/gbert-base
- xlm-roberta-base
- facebook/mbart-large-cc25
'''
