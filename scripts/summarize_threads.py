# 1: Original Post Summarization

import os.path
import os
import json
import argparse
import cv2
import colorful as cf

def main(args):
    # Get the file of threads to summarize
    # Check if there is already a summarized file for it
    path = args.f
    summarized_path = path[:len(path)-5] + "_opsummary.json"
    # path = "./datasets/redcaps/annotations/fashionadvice_2022-06_filtered_badwords.json"
    # summarized_path = "./datasets/redcaps/annotations/fashionadvice_2022-06_filtered_badwords_opsummary.json"

    with open(path, 'r') as f:
        data = json.load(f)
        threads = data['threads']
    print(f"Found {len(threads)} threads to annotate.")

    if os.path.exists(summarized_path):
        with open(summarized_path, 'r') as f:
            sum_threads = json.load(f)
            sum_threads = sum_threads['threads']
        print(f"Found {len(sum_threads)} already summarized threads. Skipping...")
        threads = threads[len(sum_threads):]
    else:
        sum_threads = list()

    # For each unsummarized thread
    for thread in threads:
        # Get the image path and open in VS Code for simplicity
        img_path = f"./datasets/redcaps/images/{thread['subreddit']}/{thread['submission_id']}.jpg"
        os.system(f"code {img_path}")


        print(cf.purple(f"{len(sum_threads)} total summaries collected so far!"))
        print(cf.cyan(f"Thread from subreddit {thread['subreddit']}:"))
        print(cf.green(thread['author']), thread['caption'])

        print(cf.purple("Enter your summary of this original post. Enter 'SKIP' to mark this to be skipped. Enter 'q' to quit."))
        summary = input()
        if summary == 'q':
            print(cf.purple("Thank you!"))
            exit()
        sum_length = len(str.split(summary))
        thread['op_summary'] = summary
        sum_threads.append(thread)
        new_data = {'info': data['info'], 'threads': sum_threads}
        with open(summarized_path, 'w') as f:
            json.dump(new_data, f, indent=4)

        # print(cf.purple(f"Summary length: {sum_length}, {int(sum_length/total_length * 100)}% of thread length."))
        print(cf.purple("Summary saved!"))
        print(cf.purple(f"{len(sum_threads)} summaries have been finished so far."))
        print(cf.purple("Enter 'c' to continue summarizing, or 'q' to quit."))
        response = input()

        if response == 'q':
            print(cf.purple("Thank you!"))
            exit()

# Helper function for displaying comments
def display_comment(comment, depth, op):
    tabs = '\t'*depth
    if comment['author'] == op:
        print(cf.green(tabs+comment['author']), comment['body'])
    else:
        print(cf.orange(tabs+comment['author']), comment['body'])
    if comment["replies"] == []:
        return len(str.split(comment['body']))
    for reply in comment["replies"]:
        return display_comment(reply, depth+1, op) + len(str.split(comment['body']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True, type=str)
    args = parser.parse_args()
    main(args)