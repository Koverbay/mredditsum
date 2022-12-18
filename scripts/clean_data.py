import json
import argparse
import csv
import os
import colorful as cf

def main(args):


    new_threads = []
    fpath = args.filepath
    if not os.path.exists(fpath):
        print(f"Oops! File not found: '{fpath}'")

    with open(fpath, 'r') as f:
        data = json.load(f)
    threads = data['threads']
    count = 1
    total = len(threads)
    for thread in threads:
        img_path = f"../thread_data/threads_to_annotate/comments_10+/images/{thread['submission_id']}.jpg"
        os.system(f"code {img_path}")
        print(f"Post {count} out of {total}:")
        print(cf.green(thread['author']), thread['caption'])
        print(f"Enter nothing to mark to be kept, 'x' to mark as unusable, 'q' to quit and save.")
        x = input()
        if x == 'q':
            print("quitting and saving now..")
            break
        if x != 'x':
            thread['usable']: 'o'
            new_threads.append(thread)
        else:
            thread['usable']: 'x'
        count += 1
        
    savefile = args.filepath[:len(args.filepath)-5]+"_clean.json"

    print(f"Writing {len(new_threads)}/{count} threads to file {savefile}...")
    write_data = {'threads': new_threads}
    with open(savefile, 'w') as f:
        json.dump(write_data,f,indent=4)
    # print(f"Writing updated data to original file:")
    # with open(args.filepath, 'w') as f:
    #     json.dump({'threads': threads},f,indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True)
    # parser.add_argument("-s", "--savefile", required=True)
    args = parser.parse_args()
    main(args)