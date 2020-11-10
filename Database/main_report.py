# Imports
import glob
import os
import collections


# Global Variables
TEXT_FILES = "C:\\Temp\\Corpus\\"
META_FILES = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"


# Methods
def read_files():
    global TEXT_FILES

    os.chdir(TEXT_FILES)
    files = glob.glob("*.txt")

    return files


def get_corpus(files):
    global TEXT_FILES

    corpus = [read_text(TEXT_FILES + file) for file in files]
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


def get_sector_distribution():
    global META_FILES

    # TODO: Use meta-files to generate a sector distribution, export graphics
    distribution = {}
    print(f"Verteilung der Datenquellen: {distribution}")


def get_document_lengths(corpus):
    # TODO: Build dictionary of document lengths and their frequencies, export graphics
    distribution =  {}
    print(f"Verteilung der Textlänge: {distribution}")


def get_words(corpus):
    # TODO: Extract all words in a non-distinct style from all documents and append each word to the list
    words = []

    return words


def get_word_distribution(corpus):
    # TODO: Setup a vocabulary, export graphics
    words = get_words(corpus)
    mcw = collections.Counter(words).most_common(100)
    print(f"Wort-Verteilung: {mcw}")


def get_n_grams(corpus, n):
    # TODO: Extract all n-grams in the corpus, export head and tail as a graphic
    distribution = {}
    print(f"Top N-Gramme: {distribution}")


# Main
def main():
    files = read_files()
    corpus = get_corpus(files)
    get_sector_distribution()
    get_document_lengths(corpus)
    get_word_distribution(corpus)
    get_n_grams(corpus, 3)


if __name__ == "__main__":
    main()
