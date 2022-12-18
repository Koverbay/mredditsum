import json
import argparse


def main(args):

    # Compare two files
    with open(args.filepath1, 'r') as f:
        data1 = json.load(f)

    with open(args.filepath2, 'r') as f:
        data2 = json.load(f)

    # Make list of id's in data1
    ids = []
    for thread in data1['threads']:
        ids.append(thread['submission_id'])
    new_threads_2 = []
    removed = []
    for thread in data2['threads']:
        id2 = thread['submission_id']
        if id2 in ids:
            print(f"Duplicate post found (id: {id2}). Removing from file {args.filepath2}...")
            removed.append(id2)
        else:
            new_threads_2.append(thread)
    
    print(f"Finished de-duplicating. {len(removed)} duplicate posts found and removed. Saving updated file to {args.savepath}...")
    with open(args.savepath, 'w') as f:
        json.dump({"threads": new_threads_2},f,indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--filepath1", required=True)
    parser.add_argument("-f2", "--filepath2", required=True)
    parser.add_argument("-s", "--savepath", required=True)
    args = parser.parse_args()
    main(args)