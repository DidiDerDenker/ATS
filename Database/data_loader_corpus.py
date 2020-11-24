# Imports
import glob
import os
import pandas as pd


# Methods
def read_text(file_path):
    with open(file_path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
        text = "\n".join(lines)
        f.close()

    return text


# Main
def main(META_FILES, LEMMATIZED_FILES):
    os.chdir(META_FILES)
    meta_files = glob.glob("*.xlsx")
    corpus = []

    for file in meta_files:
        if "tensorflow" in file:
            df = pd.read_excel(file)

            for id in df["ID"]:
                try:
                    file_path = LEMMATIZED_FILES + id + ".txt"
                    text = read_text(file_path)
                    corpus.append(text)

                except Exception as e:
                    print(e)

    return corpus


if __name__ == "__main__":
    main()
