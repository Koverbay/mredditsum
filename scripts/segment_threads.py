import os.path
import os
import json
import cv2
import colorful as cf

# Get the file of threads to segment
# Check if there is already a segmented file for it
path = "./datasets/redcaps/annotations/randomsample100.json"
summarized_path = "./datasets/redcaps/annotations/randomsample100_segmented.json"

with open(path, 'r') as f:
    threads = json.load(f)
    threads = threads['annotations']
print(f"Found {len(threads)} threads to annotate.")

if os.path.exists(summarized_path):
    with open(summarized_path, 'r') as f:
        sum_threads = json.load(f)
        sum_threads = sum_threads['annotations']
    print(f"Found {len(sum_threads)} already segmented threads. Skipping...")
    threads = threads[len(sum_threads):]
else:
    sum_threads = list()

def display_comment(comment, depth, op):
    tabs = '\t'*depth
    if comment['author'] == op:
        print(cf.pink(tabs+comment['comment_id']), cf.green(comment['author']), comment['body'])
    else:
        print(cf.pink(tabs+comment['comment_id']), cf.orange(comment['author']), comment['body'])
    # if comment["replies"] == []:
    #     len(str.split(comment['body']))
    for reply in comment["replies"]:
        display_comment(reply, depth+1, op)
        
def label_comments(comment, segment_no, split_ids):
    if comment['comment_id'] in split_ids:
        segment_no += 1
    comment['segment'] = segment_no
    for reply in comment["replies"]:
        label_comments(reply, segment_no, split_ids)

# For each unsegmented thread
for thread in threads:
    # Get the image path
    img_path = f"./datasets/redcaps/images/{thread['subreddit']}/{thread['submission_id']}.jpg"
    # Just open the image in VS Code bc idk what else to do
    os.system(f"code {img_path}")
    print(cf.purple(f"{len(sum_threads)} total threads completed so far!"))
    print(cf.cyan(f"Thread from subreddit {thread['subreddit']}:"))
    
    # total_length = len(str.split(thread['caption']))
    segment = 0
    for comment in thread['comments']:
        print(cf.green(thread['author']), thread['caption'])
        display_comment(comment, 1, thread['author'])
        print(cf.purple(f"Enter the ids of the comments where the topic changes. If there are no more topic changes, enter 'n'."))
        print(cf.purple(f"Enter 'q' to quit."))
        split_ids = list()
        while True:
            split_id = input()
            if split_id == 'n':
                break
            if split_id == 'q':
                print(cf.purple("Thank you!"))
                exit()
            else:
                split_ids.append(split_id)
        print(cf.purple(f"Split ids: {split_ids}"))
        # Go through each comment and add the segment number
        label_comments(comment, segment, split_ids)
        segment += (len(split_ids)+1)

    # print(cf.purple(f"Total number of comments: {thread['num_comments']}"))
    # print(cf.purple(f"Total length of thread: {total_length}"))
    # print(cf.purple("Enter your summary of this discussion. )
    # summary = input()
    # if summary == 'q':
    #     print(cf.purple("Thank you!"))
    #     exit()
    # sum_length = len(str.split(summary))
    # thread['k_summary'] = summary
    sum_threads.append(thread)
    data = {'annotations': sum_threads}
    with open(summarized_path, 'w') as f:
        json.dump(data, f, indent=4)

    # print(cf.purple(f"Summary length: {sum_length}, {int(sum_length/total_length * 100)}% of thread length."))
    print(cf.purple("Segment annotation saved!"))
    print(cf.purple(f"{len(sum_threads)} annotations have been finished so far."))
    print(cf.purple("Enter 'c' to continue annotating, or 'q' to quit."))
    response = input()

    if response != 'c':
        print(cf.purple("Thank you!"))
        exit()
    # Display the text (just in the terminal ig?)
    # Print submission body
        # For each top-level comment:
            # Print comment body
                # While comment.comments is not empty...
                    # Print
                    # Use helper function similar to before? include depths, also make a depth to color dictionary so we can print it all pretty like
    
    # Get user input for summary
    # After hitting enter make a "Enter 'y' to continue" kind of thing again

