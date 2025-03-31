import argparse
import os
import faiss
import sqlite3

def main(datapath:str, storepath:str):
    path_text = []
    vectors = []
    with open(filepath, "r") as read_file:
        pass



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Embeds the table data and stores within a vector database.")
    parser.add_argument('datapath')
    parser.add_argument('storepath')

    args = parser.parse_args()
    main(args.datapath, args.storepath)
