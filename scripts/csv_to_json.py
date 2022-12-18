import argparse
import csv
import ftfy
import re
import os
import json

def main(args):

    data = {}

    with open(args.filepath, 'r') as csvf:
        csvReader = csv.DictReader(csvf)

        for rows in csvReader:
            key = rows["Input.id"]
            data[key] = rows
    
    savefile = args.filepath[:(len(args.filepath)-4)] + f".json"
    with open(savefile, 'w') as jsonf:
        json.dump(data,jsonf, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True)
    # parser.add_argument("-s", "--savefile", required=True)
    parser.add_argument("-t", "--task", default="opsum", choices=["opsum",  "csum", "fsum"])
    args = parser.parse_args()
    main(args)