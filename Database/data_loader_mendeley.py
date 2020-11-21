# Imports
import os
import glob
import json
import re
import uuid
import sys
import pandas as pd

from googletrans import Translator


# User Input
JSON_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\dump_mendeley\\Part (1)\\"
META_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\data_mendeley_part_01.xlsx"


# Global Variables
TEXT_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
DATAFRAME = None
EXPORT_PROGRESS = 0


# Methods
def read_files():
    global JSON_PATH

    os.chdir(JSON_PATH)
    files = glob.glob("*.json")
    files = [JSON_PATH + file for file in files]

    return files


def initialize_dataframe():
    global META_PATH
    global DATAFRAME

    if os.path.exists(META_PATH):
        DATAFRAME = pd.read_excel(META_PATH)

    else:
        DATAFRAME = pd.DataFrame(data=[], columns=["ID", "Title"])


def update_dataframe(text):
    global DATAFRAME
    global EXPORT_PROGRESS

    if len(text) > 1000:
        id = str(uuid.uuid4()).upper()
        df = pd.DataFrame(data=[[id, "-"]], columns=["ID", "Title"])
        DATAFRAME = pd.concat([DATAFRAME, df], axis=0)
        export_text(id, text)

        sys.stdout.write("\rFile %i..." % (EXPORT_PROGRESS + 1))
        sys.stdout.flush()
        EXPORT_PROGRESS += 1


def export_dataframe():
    global META_PATH
    global DATAFRAME

    DATAFRAME = DATAFRAME.sort_values(by=["Title"])
    DATAFRAME[["ID", "Title"]].to_excel(META_PATH, index=False)


def export_text(id, text):
    global TEXT_PATH

    file_path = TEXT_PATH + id + ".txt"

    try:
        with open(file_path, "w", encoding="utf8") as f:
            f.write(text)

    except Exception as e:
        print(e)


def translate_sentence(translator, sentence):
    translation = None

    try:
        translation = translator.translate(sentence, src="EN", dest="DE")

    except Exception as e:
        pass

    if translation is not None:
        sentence = re.sub("\s*\(.*?\)\s*", " ", translation.text)
        sentence = sentence.replace(" .", ".")
        return sentence

    else:
        return ""


def process_files(files):
    global EXPORT_PROGRESS

    for file in files:
        with open(file) as f:
            translator = Translator() # TODO
            data = json.load(f)
            sentences = [translate_sentence(translator, item["sentence"]) for item in data["body_text"]]
            sentences = filter(None, sentences)
            text = "\n".join(sentences)
            update_dataframe(text)

            if EXPORT_PROGRESS % 500 == 0:
                export_dataframe()


# Main
def main():
    files = read_files()
    initialize_dataframe()
    process_files(files)
    export_dataframe()


if __name__ == "__main__":
    main()
