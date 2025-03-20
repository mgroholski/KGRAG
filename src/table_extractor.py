import json
import argparse
from bs4 import BeautifulSoup as bs
import os

parser = argparse.ArgumentParser(description="Extracts tables from html.")
parser.add_argument('filepath')
args = parser.parse_args()

with open(args.filepath, "r") as file:
    with open(os.path.expanduser(f"data/filtered_{os.path.basename(args.filepath)}"), "w") as write_file:
        for idx, line in enumerate(file):
            line_json = json.loads(line)
            line_html = line_json['document_html']
            soup = bs(line_html, "lxml")

            def build_json(t)->dict:
                data = {}
                curObj = None
                for row in t.find_all("tr", recursion=False):
                    # Check if row is subtable
                    subtable = row.find("table", recursion=False)
                    heading = row.find("th", recursion=False)
                    row_data = row.find_all("td", recursion=False)
                    if subtable:
                        if curObj == None:
                            raise Exception(f"curObj not initialized and calling recursive: {row}")
                        curObj.update(build_json(subtable))
                    elif heading:
                        curObj = {}
                        data[heading.get_text()] = curObj
                    elif len(row_data) == 1:
                        if curObj == None:
                            raise Exception(f"Single data before curObj is initialized: {row}")
                        objTitle = row_data[0].get_text()
                        curObj[objTitle] = {}
                        curObj = curObj[objTitle]
                    elif len(row_data) == 2:
                        if curObj == None:
                            raise Exception(f"Adding data before curObj is initialized: {row}")
                        curObj[row_data[0].get_text()] = row_data[1].get_text()
                    else:
                        raise Exception(f"Invalid data row length: {row}")

                return data

            tables = soup.find_all("table", recursion=False)
            for table_idx, table in enumerate(tables):
                    try:
                        write_file.write(json.dumps(build_json(table)) + "\n")
                    except Exception as e:
                        print(f"Skipping table {table_idx + 1} on line {idx + 1} because {e}.")
