import os.path
import os
import json
import cv2
import colorful as cf
import random
#cf.use_style('solarized')

# Recursive helper function for displaying comments
def display_comment(comment, depth, op):
    tabs = '\t'*depth
    if comment['author'] == op:
        print(cf.green(tabs+comment['author']), comment['body'])
    else:
        print(cf.orange(tabs+comment['author']), comment['body'])
    for reply in comment["replies"]:
        display_comment(reply, depth+1, op)

# Read in all subreddits
# subreddits = list()
# # with open("../subreddits.txt", 'r') as f:
# #     subreddits = [line.rstrip() for line in f]
# subreddits.append("r/learnart")



subreddit = "r/learnart"

# subreddits.append("r/mildlyinteresting")
# subreddits.append("r/damnthatsinteresting")
# # subreddits.append("r/eyebleach")
# # subreddits.append("r/itookapicture")
# subreddits.append("r/perfectfit")
# subreddits.append("r/somethingimade")
# subreddits.append("r/thedepthsbelow")
# subreddits.append("r/upcycling")
# subreddits.append("r/zerowaste")



# Get a random subreddit to sample from
print(cf.purple(f"Sampling Reddit threads for browsing..."))
while True:
    # subreddit = subreddits[random.randint(0, len(subreddits)-1)][1:]
    # print(cf.purple(f"Sampled thread from subreddit {subreddit}:"))

    years = ['2019', '2020', '2021']
    months = ['01', '02','03','04','05','06','07','08','09','10','11','12']

    year = years[random.randint(0,2)]
    month=months[random.randint(0,11)]

    path = f"./datasets/redcaps/annotations_5+comments/{subreddit[2:]}_{year}-{month}_filtered.json"

    with open(path, 'r') as f:
        data = json.load(f)
        threads = data['threads']
    if len(threads) == 0:
        continue
    thread = threads[random.randint(0, len(threads)-1)]

    img_path = f"./datasets/redcaps/images/{thread['subreddit']}/{thread['submission_id']}.jpg"
    os.system(f"code {img_path}")
    print(cf.green(thread['author']), thread['caption'])
    for comment in thread['comments']:
        display_comment(comment, 1, thread['author'])

    print(cf.purple("Hit 'enter' to see another thread. Enter 's' to save this thread to starred threads. Enter 'q' to quit."))
    i = input()
    if i == 'q':
        exit()
    if i == 's':
        save_path = f"./datasets/redcaps/annotations/starred.json"
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                saved_threads = json.load(f)    
        else:
            saved_threads = list()
        saved_threads.append(thread)
        with open(save_path, 'w') as f:
            json.dump(saved_threads, f, indent=4)
        print(cf.purple("Saved thread to starred threads. Continuing..."))



# # For each unsummarized thread
# for thread in threads:
#     # Get the image path
#     img_path = f"./datasets/redcaps/images/{thread['subreddit']}/{thread['submission_id']}.jpg"
#     # Just open the image in VS Code bc idk what else to do
#     os.system(f"code {img_path}")
#     print(cf.purple(f"{len(sum_threads)} total summaries collected so far!"))
#     print(cf.cyan(f"Thread from subreddit {thread['subreddit']}:"))
#     print(cf.green(thread['author']), thread['caption'])
#     total_length = len(str.split(thread['caption']))
#     for comment in thread['comments']:
#         total_length += display_comment(comment, 1, thread['author'])

    

#     print(cf.purple(f"Total number of comments: {thread['num_comments']}"))
#     print(cf.purple(f"Total length of thread: {total_length}"))
#     print(cf.purple("Enter your summary of this discussion. Enter 'SKIP' to mark this to be skipped. Enter 'q' to quit."))
#     summary = input()
#     if summary == 'q':
#         print(cf.purple("Thank you!"))
#         exit()
#     sum_length = len(str.split(summary))
#     thread['k_summary'] = summary
#     sum_threads.append(thread)
#     data = {'annotations': sum_threads}
#     with open(summarized_path, 'w') as f:
#         json.dump(data, f, indent=4)

#     print(cf.purple(f"Summary length: {sum_length}, {int(sum_length/total_length * 100)}% of thread length."))
#     print(cf.purple("Summary saved!"))
#     print(cf.purple(f"{len(sum_threads)} summaries have been finished so far."))
#     print(cf.purple("Enter 'c' to continue summarizing, or 'q' to quit."))
#     response = input()

#     if response != 'c':
#         print(cf.purple("Thank you!"))
#         exit()
#     # Display the text (just in the terminal ig?)
#     # Print submission body
#         # For each top-level comment:
            # Print comment body
                # While comment.comments is not empty...
                    # Print
                    # Use helper function similar to before? include depths, also make a depth to color dictionary so we can print it all pretty like
    
    # Get user input for summary
    # After hitting enter make a "Enter 'y' to continue" kind of thing again

