# Imports
import os
import glob
import re
import string
import nltk
import pandas as pd


# Global Variables
TEXT_FILES = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
META_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"


# Methods
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^a-zäöüA-ZÄÖÜß]", " ", text)

    return text


# Main
def main():
    global TEXT_FILES
    global META_PATH

    token_count = 0
    vocab = {}

    os.chdir(META_PATH)
    meta_files = glob.glob("*.xlsx")

    for file in meta_files:
        corpus = ""

        if "wikipedia" in file:
            print("Starting iteration...")
            df = pd.read_excel(file)

            for id in df["ID"]:
                file_path = TEXT_FILES + id + ".txt"

                with open(file_path, "r", encoding="utf8", errors="ignore") as f:
                    lines = f.readlines()
                    text = " ".join(lines)
                    corpus = corpus + " " + text
                    f.close()

            corpus = clean_text(corpus)
            tokens = [token.lower() for token in nltk.word_tokenize(corpus)]
            freq = nltk.FreqDist(tokens)
            df = pd.DataFrame(freq.items(), columns=["Token", "Freq"])
            token_count += sum(list(df["Freq"]))

            for index, row in df.iterrows():
                token = row["Token"]
                freq = row["Freq"]

                if token not in vocab.keys():
                    vocab[token] = freq

                else:
                    vocab[token] += freq

            print(f"Finished iteration ({file})")

    print(f"Distinct token: {len(vocab.keys())}")
    print(f"Total token: {token_count}")

    df = pd.DataFrame.from_dict(vocab, orient="index")
    df.to_excel("C:\\Temp\\Types.xlsx")

    # TODO: Send report again
    # TODO: Remove this script


if __name__ == "__main__":
    main()
