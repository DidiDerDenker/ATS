# Imports
import os
import glob
import json
import pandas as pd


# Global Variables
JSON_PATH = "\\\\NAS-SYSTEM\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\dump_mendeley\\"


# Methods


# Main
os.chdir(JSON_PATH)
files = glob.glob("*.json")
files = [JSON_PATH + file for file in files]

for file in files:
    with open(file) as f:
        data = json.load(f)
        sentences = [item["sentence"] for item in data["body_text"]]
        text = "\n".join(sentences)

        if len(text) > 1000:
            print(text)
            print(len(text))

        exit()
