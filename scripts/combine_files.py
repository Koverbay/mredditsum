import json
import argparse
import csv
import os

def main(args):

    subreddits = list()
    with open("../subreddits_to_write.txt", 'r') as f:
        subreddits = [line.rstrip() for line in f]
    new_threads = []
    for sub in subreddits:
        fpath = args.folderpath + f"/{sub[2:]}_clustered_anon.json"
        if not os.path.exists(fpath):
            print(f"Oops! File not found: '{fpath}'")
            continue
        print(f"Getting data for subreddit {sub}...")
        with open(fpath, 'r') as f:
            data = json.load(f)
        threads = data['threads']
        for thread in threads:
            new_threads.append(thread)

    print(f"Writing all threads to file {args.savefile}...")
    write_data = {'threads': new_threads}
    with open(args.folderpath + '/' + args.savefile, 'w') as f:
        json.dump(write_data,f,indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folderpath", required=True)
    parser.add_argument("-s", "--savefile", required=True)
    args = parser.parse_args()
    main(args)