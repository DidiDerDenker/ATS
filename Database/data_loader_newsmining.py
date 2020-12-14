# Imports
import pyodbc
import pandas as pd
import asyncio
import time
import uuid

from pyppeteer import launch


# Global Variables
TEXT_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\text_files\\"
SUMMARY_PATH = "B:\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\summary_files\\"
META_PATH = "B:\\home\\CloudStation\\Drive\\Server [Daniel]\\Active\\[Karriere]\\Organisationen\\Data Science\\AutomaticTextSummarization\\Database\\meta_files\\data_newsmining.xlsx"


# Methods
def sql_get_data():
    conn = pyodbc.connect("Driver={SQL Server};"
                          "Server=VOGEL-PC\\SQL_Vogel;"
                          "Database=NewsMining;"
                          "Trusted_Connection=yes;")
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM NewsMining WHERE Agency = 'Spiegel'", conn) # TODO: Use all agencies

    conn.commit()
    cursor.close()
    conn.close()

    return df


async def crawl_text(url):
    browser = await launch(headless=False, executablePath="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe")
    page = await browser.newPage()

    await page.goto(url)
    await page.click("body.iv > div.message > div.message-component.message-row > button") # TODO: Accept cookies
    element = await page.querySelector("article")
    content = await page.evaluate("(element) => element.innerText", element) # TODO: Check content
    await browser.close()

    return content


def export_meta_file(log):
    global META_PATH

    df = pd.DataFrame(data=log, columns=["ID", "Title"])
    df = df.sort_values(by=["Title"])
    df[["ID", "Title"]].to_excel(META_PATH, index=False)
    print("Export done...")


# Main
def main():
    df = sql_get_data()
    log = []

    for index, row in df.iterrows():
        summary = row["Summary"]
        text = asyncio.get_event_loop().run_until_complete(crawl_text(row["URL"]))

        id = str(uuid.uuid4()).upper()
        text_path = TEXT_PATH + id + ".txt"
        summary_path = SUMMARY_PATH + id + ".txt"

        print(summary)
        print(text)
        exit() # TODO: Test and use

        try:
            with open(text_path, "w", encoding="utf8") as f:
                f.write(text)
                f.close()

            with open(summary_path, "w", encoding="utf8") as f:
                f.write(summary)
                f.close()

            log.append((id, "-"))

        except Exception as e:
            print(e)

    export_meta_file(log)


if __name__ == "__main__":
    main()
