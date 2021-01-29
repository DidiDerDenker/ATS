# Imports
import wget
import os
import tarfile
import gzip
import zipfile
import argparse


# Notebook
parser = argparse.ArgumentParser()
parser.add_argument("--glove", action="store_true")
args = parser.parse_args()


with tarfile.open("summary.tar.gz", "r:gz") as tar:
    tar.extractall()


with gzip.open("sumdata/train/train.article.txt.gz", "rb") as gz:
    with open("sumdata/train/train.article.txt", "wb") as out:
        out.write(gz.read())


with gzip.open("sumdata/train/train.title.txt.gz", "rb") as gz:
    with open("sumdata/train/train.title.txt", "wb") as out:
        out.write(gz.read())


if args.glove:
    glove_dir = "glove"
    glove_url = "https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip"

    if not os.path.exists(glove_dir):
        os.mkdir(glove_dir)

    wget.download(glove_url, out=glove_dir)

    with zipfile.ZipFile(os.path.join("glove", "glove.42B.300d.zip"), "r") as z:
        z.extractall(glove_dir)
