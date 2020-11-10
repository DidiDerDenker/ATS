# Imports
import pandas as pd
import sys
import uuid

from bs4 import BeautifulSoup


# User Input
DUMP_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\json_dump\\part (1).json"
META_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\data_open_legal_part_1.xlsx"


# Global Variables
TEXT_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
DATAFRAME = None
EXPORT_PROGRESS = 0


# Methods
def clean_text(raw_text):
    parts = raw_text.split("\n")
    text = ""

    for part in parts:
        word_count = len(part.split())

        if word_count is not None and word_count > 1:
            if part[0].isdigit():
                part = " ".join(part.split()[1:])

            text += part + "\n"

    return text.strip()


def export_text(id, text):
    global TEXT_PATH
    global EXPORT_PROGRESS

    file_path = TEXT_PATH + id + ".txt"

    try:
        with open(file_path, "w", encoding="utf8") as f:
            f.write(text)

    except Exception as e:
        print(e)

    sys.stdout.write("\rFile %i..." % (EXPORT_PROGRESS + 1))
    sys.stdout.flush()
    EXPORT_PROGRESS += 1


# Main
def main():
    global DUMP_PATH
    global DATAFRAME

    df_json = pd.read_json(DUMP_PATH, lines=True)

    for index, row in df_json.iterrows():
        soup = BeautifulSoup(row["content"], features="html.parser")
        raw_text = soup.get_text().strip()
        text = clean_text(raw_text)

        if len(text) > 1000:
            id = str(uuid.uuid4()).upper()
            df = pd.DataFrame(data=[[id]], columns=["ID"])
            DATAFRAME = pd.concat([DATAFRAME, df], axis=0)
            export_text(id, text)

    DATAFRAME[["ID"]].to_excel(META_PATH, index=False)


if __name__ == "__main__":
    main()
