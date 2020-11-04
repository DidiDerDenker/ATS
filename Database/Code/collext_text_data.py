# Imports
import os
import sys
import uuid
import wikipediaapi as wp


# Paths
STOP_SECTIONS = "./stop_sections.txt"
WIKIPEDIA_CATEGORIES = "./wikipedia_categories.txt"
OUTPUT_DIR = "C:\\Temp\\ATS\\Database\\"
LIMIT_PER_CATEGORY = 100 # TODO: Increase limit
CATEGORY_COUNT = 0
EXPORT_PROGRESS = 0


# Methods
def get_lines_from_txt(file_name):
    with open(file_name, "r") as f:
        names = f.readlines()
        names = [name.replace("\n", "") for name in names]

    return names


def get_page_text(article_page, stop_sections):
    sections = article_page.sections
    text = article_page.title

    if len(sections) > 0:
        for section in sections:
            if section.title not in stop_sections:
                text = text + "\n\n" + section.text

    return clean_text(text)


def clean_text(text):
    # TODO: Clean text

    return text


def export_text(path, text):
    global EXPORT_PROGRESS

    file_name = path + str(uuid.uuid4()).upper() + ".txt"

    # TODO: Fix codec error and then remove lines after check_text
    with open(file_name, "w") as f:
        try:
            f.write(text)

        except Exception as e:
            print(e)

    check_text = None

    with open(file_name, "r") as f:
        check_text = f.readlines()

    if len(check_text) < 3:
        os.remove(file_name)

    else:
        sys.stdout.write("\rFile %i..." % (EXPORT_PROGRESS + 1))
        sys.stdout.flush()
        EXPORT_PROGRESS += 1


def analyze_category(wiki_engine, stop_sections, category):
    global OUTPUT_DIR
    global LIMIT_PER_CATEGORY
    global CATEGORY_COUNT

    try:
        category_page = wiki_engine.page("Category:" + str(category))

        while CATEGORY_COUNT < LIMIT_PER_CATEGORY:
            for member in category_page.categorymembers:
                if "Kategorie:" in member:
                    analyze_category(wiki_engine, stop_sections, member.replace("Kategorie:", ""))

                else:
                    article_page = wiki_engine.page(member)

                    if article_page.exists():
                        # TODO: Append first row of text to a list, later check if title exists already
                        text = get_page_text(article_page, stop_sections)

                        if len(text) > 500:
                            export_text(OUTPUT_DIR, text)
                            CATEGORY_COUNT += 1

    except Exception as e:
        print(e)


# Main
def main():
    global WIKIPEDIA_CATEGORIES
    global STOP_SECTIONS
    global CATEGORY_COUNT

    wiki_engine = wp.Wikipedia(language="de", extract_format=wp.ExtractFormat.WIKI)
    wikipedia_categories = get_lines_from_txt(WIKIPEDIA_CATEGORIES)
    stop_sections = get_lines_from_txt(STOP_SECTIONS)

    for category in wikipedia_categories:
        CATEGORY_COUNT = 0
        analyze_category(wiki_engine, stop_sections, category)


if __name__ == "__main__":
    main()
