# 2: Clustering
import os.path
import os
import json
import argparse
import cv2
import colorful as cf

def main(args):
    # path = "./datasets/redcaps/annotations/fashionadvice_2022-06_filtered_badwords_opsummary.json"
    # clustered_path = "./datasets/redcaps/annotations/fashionadvice_2022-06_filtered_badwords_clustered.json"

    path = args.f
    clustered_path = path[:len(path)-5] + "_clustered.json"
    # prev_path = path[:len(path)-5] _ "_opsummary.json"

    with open(path, 'r') as f:
        data = json.load(f)
        threads = data['threads']
    print(f"Found {len(threads)} threads to annotate.")

    # Load existing annotations
    if os.path.exists(clustered_path):
        with open(clustered_path, 'r') as f:
            clustered_threads = json.load(f)
            if len(clustered_threads) != 0 :
                clustered_threads = clustered_threads['threads']
        print(f"Found {len(clustered_threads)} already clustered threads. Skipping...")
        threads = threads[len(clustered_threads):]
    else:
        clustered_threads = list()

    # For each unsegmented thread
    for thread in threads:
        # Get the image path
        img_path = f"./datasets/redcaps/images/{thread['subreddit']}/{thread['submission_id']}.jpg"
        # Just open the image in VS Code bc idk what else to do
        os.system(f"code {img_path}")
        print(cf.purple(f"{len(clustered_threads)} total threads completed so far!"))
        print(cf.cyan(f"Thread from subreddit {thread['subreddit']}:"))
        
        # Check each top-level comment and assign a cluster
        print(cf.green(thread['author']), thread['caption'])
        for comment in thread['comments']:
            print(cf.orange('\t' + comment['author']), comment['body'])
            print(cf.purple(f"Enter cluster number to assign this comment to."))
            print(cf.purple(f"Enter 'q' to quit."))
            
            cluster = input()
            if cluster == 'q':
                print(cf.purple("Thank you!"))
                exit()
            cluster = int(cluster)
        
            print(cf.purple(f"Adding comment to cluster {cluster}."))
            # Go through each comment and add the cluster number
            label_comments(comment, cluster)
            

        clustered_threads.append(thread)
        new_data = {'info': data['info'], 'threads': clustered_threads}
        with open(clustered_path, 'w') as f:
            json.dump(new_data, f, indent=4)

        print(cf.purple("Cluster annotation saved!"))
        print(cf.purple(f"{len(clustered_threads)} annotations have been finished so far."))
        print(cf.purple("Enter 'q' to quit. Hit enter to continue annotating."))
        response = input()

        if response == 'q':
            print(cf.purple("Thank you!"))
            exit()
    
def label_comments(comment, cluster_no):
    comment['cluster'] = cluster_no
    for reply in comment["replies"]:
        label_comments(reply,cluster_no)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', required=True, type=str)
    args = parser.parse_args()
    main(args)