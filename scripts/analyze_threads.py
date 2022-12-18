# Go through each subreddit by category and calculate how many posts there are, the average comment length, um...yea.
import json
import argparse

def main(args):
    subredditfile = args.subredditfile
    threadfolder = args.threadfolder
    counts = {}

    with open(subredditfile, 'r') as f:
        subreddits = json.load(f)

    # for each category, check each subreddit and compute it
    for category in subreddits:
        num_posts = 0
        total_comments = 0
        total_comment_length = 0
        subreddits_by_cat = subreddits[category]
        for subreddit in subreddits_by_cat:
            filepath = threadfolder + "/" + subreddit + "_2022-01.json"
            with open(filepath, 'r') as f:
                sub_data = json.load(f)
            num_posts += len(sub_data["annotations"])
            
            for post in sub_data["annotations"]:
                total_comments += post["num_comments"]
                total_comment_length += getTotalCommentLength(post)
            if num_posts != 0 :
                avg_num_comments = total_comments / num_posts
                
            else:
                avg_num_comments = 0
            if total_comment_length != 0:
                avg_comment_length = total_comment_length / total_comments
            else:
                avg_comment_length = 0

        counts[category] = {
            "num_posts": num_posts,
            "avg_num_comments": avg_num_comments,
             "avg_comment_length": avg_comment_length,
        }
    print(counts)

def getTotalCommentLength(post):
    total_length = len(post['caption'].split())
    for reply in post['comments']:
        total_length += getTotalCommentLengthComments(reply)
    return total_length

def getTotalCommentLengthComments(comment):
    if comment['replies'] == []:
        return len(comment['body'].split())
    else:
        total = len(comment['body'].split())
        for reply in comment['replies']:
            total += getTotalCommentLengthComments(reply)
        return total



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--subredditfile', type=str, default="../subreddits.json")
    parser.add_argument('--threadfolder', type=str, default="datasets/redcaps/annotations_2022-01")

    args = parser.parse_args()
    main(args)