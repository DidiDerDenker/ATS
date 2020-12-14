# Imports
import json
import uuid
import pandas as pd


# Global Variables
TEXT_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
SUMMARY_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\summary_files\\"
META_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\"


# Methods
def export_meta_file(name, log):
    global META_PATH

    meta_path = META_PATH + name + ".xlsx"
    df = pd.DataFrame(data=log, columns=["ID", "Title"])
    df = df.sort_values(by=["Title"])
    df[["ID", "Title"]].to_excel(meta_path, index=False)
    print("Export done...")


# Main
def main():
    global TEXT_PATH
    global SUMMARY_PATH
    global META_PATH

    temp_file = "C:\\Users\\didid\\Downloads\\TLDR-TEMP\\tldr.jsonl" # TODO: Remove file afterwards from directory, update this path
    log = []
    iteration_num = 0
    export_num = 1

    for line in open(temp_file, encoding="utf8"):
        d = json.loads(line)
        text = d["content"]
        summary = d["summary"]

        id = str(uuid.uuid4()).upper()
        text_path = TEXT_PATH + id + ".txt"
        summary_path = SUMMARY_PATH + id + ".txt"

        try:
            with open(text_path, "w", encoding="utf8") as f:
                f.write(text)
                f.close()

            with open(summary_path, "w", encoding="utf8") as f:
                f.write(summary)
                f.close()

            log.append((id, "-"))
            iteration_num += 1

        except Exception as e:
            print(e)

        if iteration_num % 100000 == 0:
            name = "data_tldr_" + str(export_num)
            export_meta_file(name, log)
            export_num += 1
            log = []

    name = "data_tldr_" + str(export_num)
    export_meta_file(name, log)


if __name__ == "__main__":
    main()
