# Documentation
'''
https://pypi.org/project/Wikipedia-API/
'''


# Imports
import pandas as pd
import wikipediaapi as wp
import os
import sys
import uuid


# User Input
INITIAL_CATEGORY = "Biologie"
MIN_LENGTH = 1000
LIMIT = 150000


# Global Variables
EXPORT_PATH_TEXT = "./text_files\\"
EXPORT_PATH_META = "./meta_files\\data_wikipedia_" + str(INITIAL_CATEGORY).lower() + ".xlsx"
STOP_SECTIONS = "./stop_sections.txt"
DATAFRAME = None
ITERATION_NUM = 0
EXPORT_PROGRESS = 0


# Methods
def get_stop_sections(file_name):
    with open(file_name, "r") as f:
        names = f.readlines()
        names = [name.replace("\n", "") for name in names]

    return names


def get_page_text(article_page, stop_sections):
    sections = article_page.sections
    text = str()

    if len(sections) > 0:
        for section in sections:
            if section.title not in stop_sections:
                text = text + "\n\n" + clean_text(section.text)

    return text


def clean_text(text):
    text = text.split("== Einzelnachweise ==")[0]

    return text


def initialize_dataframe():
    global EXPORT_PATH_META
    global DATAFRAME

    if os.path.exists(EXPORT_PATH_META):
        DATAFRAME = pd.read_excel(EXPORT_PATH_META)

    else:
        DATAFRAME = pd.DataFrame(data=[], columns=["ID", "Title"])


def update_dataframe(title, text):
    global MIN_LENGTH
    global DATAFRAME
    global ITERATION_NUM
    global EXPORT_PROGRESS

    if len(text) > MIN_LENGTH and title not in list(DATAFRAME["Title"].values):
        id = str(uuid.uuid4()).upper()
        df = pd.DataFrame(data=[[id, title]], columns=["ID", "Title"])
        DATAFRAME = pd.concat([DATAFRAME, df], axis=0)
        export_text(id, text)

        sys.stdout.write("\rFile %i..." % (EXPORT_PROGRESS + 1))
        sys.stdout.flush()
        ITERATION_NUM += 1
        EXPORT_PROGRESS += 1


def export_dataframe():
    global EXPORT_PATH_META
    global DATAFRAME

    DATAFRAME = DATAFRAME.sort_values(by=["Title"])
    DATAFRAME[["ID", "Title"]].to_excel(EXPORT_PATH_META, index=False)


def export_text(id, text):
    global INITIAL_CATEGORY
    global EXPORT_PATH_TEXT

    file_path = EXPORT_PATH_TEXT + id + ".txt"

    try:
        with open(file_path, "w", encoding="utf8") as f:
            f.write(text)

    except Exception as e:
        print(e)


def analyze_category(wiki_engine, stop_sections, category):
    global LIMIT
    global ITERATION_NUM

    try:
        print(f"Kategorie: {category}")
        category_page = wiki_engine.page("Category:" + str(category))

    except Exception as e:
        print(e)

    for member in category_page.categorymembers:
        if ITERATION_NUM % 1000 == 0 or ITERATION_NUM == LIMIT:
            export_dataframe()

        if ITERATION_NUM > LIMIT:
            break

        if "Kategorie:" in member:
            analyze_category(wiki_engine, stop_sections, member.replace("Kategorie:", ""))

        else:
            article_page = wiki_engine.page(member)

            if article_page.exists():
                text = get_page_text(article_page, stop_sections)
                update_dataframe(article_page.title, text[2:len(text)])


# Main
def main():
    global INITIAL_CATEGORY
    global STOP_SECTIONS

    wiki_engine = wp.Wikipedia(language="de", extract_format=wp.ExtractFormat.WIKI)
    stop_sections = get_stop_sections(STOP_SECTIONS)
    initialize_dataframe()
    analyze_category(wiki_engine, stop_sections, INITIAL_CATEGORY)


if __name__ == "__main__":
    main()