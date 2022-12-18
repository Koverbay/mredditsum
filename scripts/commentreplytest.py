import praw
import json
from typing import Any, Dict, List

def getReplies(comment):
    if len(comment.replies) == 0:
        desc = List[Dict[str, Any]]
        desc = [{
                    "submission_id": comment.id,
                    "body": comment.body,
                    "score": comment.score,
                    "author": comment.author,
                    "created_utc": int(comment.created_utc),
                    "permalink": comment.permalink,
                    "is_submitter": comment.is_submitter,
                    "replies": [],
                }]
        return desc
    else:
        # if there are replies...
        replylist = []
        for reply in comment.replies:
            replylist.append(getReplies(reply))
        desc = List[Dict[str, Any]]
        desc = [{
                    "submission_id": comment.id,
                    "body": comment.body,
                    "score": comment.score,
                    "author": comment.author,
                    "created_utc": int(comment.created_utc),
                    "permalink": comment.permalink,
                    "is_submitter": comment.is_submitter,
                    "replies": replylist,
                }]
        return desc


print("Loading reddit...")
credentials = json.load(open("././credentials.json"))
reddit = praw.Reddit(
    client_id=credentials["reddit"]["client_id"],
    client_secret=credentials["reddit"]["client_secret"],
    user_agent=credentials["reddit"]["user_agent"],
)

print("Retrieving submission...")
id = "t3_rv1cj6"
submissions = reddit.info([id])
"Submission retrieved!"
for submission in submissions:
    "Checking comments..."
    comments = submission.comments
    jsontree = [] #List[Dict[str, Any]]

    for comment in comments:
        replytree = getReplies(comment)
        print(replytree)
        print("Next...")
        jsontree.extend(replytree)
    print(jsontree)