# Imports
import glob
import os
import collections


# Global Variables
PATH = "C:\\Temp\\Corpus\\"


# Methods
def read_files():
    global PATH

    os.chdir(PATH)
    files = glob.glob("*.txt")

    return files


def get_corpus(files):
    corpus = [read_text(PATH + file) for file in files]
    print(f"Anzahl der Textdateien: {len(corpus)}")
    print(corpus[:3])

    return corpus


def read_text(path):
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
        text = "\n".join(lines)
        text = clean_text(text)

    return text


def clean_text(text):
    # TODO: Check and clean texts properly

    '''
    text = text.str.lower()
    text = text.str.replace(r"@", "")
    text = text.str.replace(r"[^A-Za-z0-9öäüÖÄÜß()!?]", " ")
    text = text.str.replace("\s{2,}", " ")
    '''

    return text


def get_document_lengths(corpus):
    # TODO: Build dictionary of document lengths and their frequencies
    distribution =  {}
    print(f"Verteilung der Textlänge: {distribution}")


def get_words(corpus):
    # TODO: Extract all words in a non-distinct style from all documents and append each word to the list
    words = []

    return words


def get_word_distribution(corpus):
    words = get_words(corpus)
    mcw = collections.Counter(words).most_common(100)
    print(f"Wort-Verteilung: {mcw}")


# Main
def main():
    files = read_files()
    corpus = get_corpus(files)
    get_document_lengths(corpus)
    get_word_distribution(corpus)


if __name__ == "__main__":
    main()
