# 3: Cluster Ranking
# 4: Cluster Summarization
# 5: Summary Synthesis

import os.path
import os
import json
import argparse
import cv2
import colorful as cf




# Get the file of threads to summarize
# Check if there is already a summarized file for it
# path = "./datasets/redcaps/annotations/fashionadvice_2022-06_filtered_badwords_clustered.json"
# summarized_path = "./datasets/redcaps/annotations/fashionadvice_2022-06_filtered_badwords_finalsummary.json"

def main(args):

    path = args.f
    summarized_path = path[:len(path)-5] + "_finalsummary.json"
    # prev_path = path[:len(path)-5] _ "_clustered.json"


    # Get threads to annotate
    with open(path, 'r') as f:
        threads = json.load(f)
        threads = threads['threads']
    print(f"Found {len(threads)} threads to annotate.")

    # Check if we have summarized some already
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
        # Get all the clusters of comments, together with their total score
        # [[score, [top-level-comment1, top-level-comment2]], ...]
        clusters = get_clusters(thread)

        # Get the image path and open in VS Code for simplicity
        img_path = f"./datasets/redcaps/images/{thread['subreddit']}/{thread['submission_id']}.jpg"
        os.system(f"code {img_path}")

        final_summary = thread['op_summary']

        print(cf.purple(f"{len(sum_threads)} total summaries collected so far!"))
        print(cf.cyan(f"Thread from subreddit {thread['subreddit']}:"))
        print(cf.green(thread['author']), thread['caption'])

        cluster_summaries = {}
        # Limit to top 5 clusters
        for cluster in clusters[:5]:
            for comment in cluster['comments']:
                display_comment(comment, 1)
            print(cf.purple("Enter your summary of this cluster of comments. Enter 'q' to quit."))
            summary = input()
            if summary == 'q':
                print(cf.purple("Thank you!"))
                exit()
            cluster['summary'] = summary

            # Add space and each cluster summary to the final summary
            final_summary += ' '
            final_summary += summary

        thread['clusters'] = clusters
        thread['final_summary'] = final_summary
        sum_threads.append(thread)
        data = {'threads': sum_threads}
        with open(summarized_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(cf.purple("Cluster summaries saved!"))
        print(cf.purple(f"{len(sum_threads)} summaries have been finished so far."))
        print(cf.purple("Enter 'q' to quit, or press any key to continue summarizing."))
        response = input()

        if response == 'q':
            print(cf.purple("Thank you!"))
            exit()

# Helper function for displaying comments
def display_comment(comment, depth):
    tabs = '\t'*depth
    if comment['is_submitter'] == True:
        print(cf.green(tabs+comment['author']), comment['body'])
    else:
        print(cf.orange(tabs+comment['author']), comment['body'])
    if comment["replies"] == []:
        return len(str.split(comment['body']))
    for reply in comment["replies"]:
        return display_comment(reply, depth+1) + len(str.split(comment['body']))

def get_total_score(comment):
    score = comment['score']
    for reply in comment['replies']:
        score += get_total_score(reply)
    return score

def get_clusters(thread):
    clusters = []
    comments = thread['comments']
    num_clusters = max(comment['cluster'] for comment in comments)
    cluster_no = 1
    while cluster_no <= num_clusters:
        cluster_comments = []
        cluster_score = 0
        for tl_comment in comments:
            if tl_comment['cluster'] == cluster_no:
                cluster_comments.append(tl_comment)
                score = get_total_score(tl_comment)
                cluster_score += score
        cluster = {
            'cluster_no': cluster_no,
            'cluster_score': cluster_score,
            'comments': cluster_comments
            }
        clusters.append(cluster)
        cluster_no += 1
    clusters.sort(reverse=True, key=lambda i: i['cluster_score'])
    return clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', required=True, type=str)
    args = parser.parse_args()
    main(args)