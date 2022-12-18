import argparse
import json
import re
import urllib.request

# GitHub repository for NSFW words
WORDS_REPO: str = "LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words"
# RegEx for Finding URL's
regex = '(?:(https?|s?ftp):\\/\\/)(?:www\\.)?((?:(?:[A-Z0-9][A-Z0-9-]{0,61}[A-Z0-9]\\.)+)([A-Z]{2,6})|(?:\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}))(?::(\\d{1,5}))?(?:(\\/\\S+)*)'
url_finder = re.compile(regex, re.IGNORECASE)

def main(args):
    filename = args.filename
    savefile = args.savefile
    maxcomments = args.maxcomments
    filtertype = args.filter
    with open(filename, 'r') as f:
        threads_data = json.load(f)
        threads = threads_data['threads']
    new_threads_data = threads_data
    new_threads = []

    # Read the blocklist of 'bad' words from the Github repo.
    blockwords_file = urllib.request.urlopen(
        f"https://raw.githubusercontent.com/{WORDS_REPO}/master/en"
    )
    blockwords: List[str] = [
        line.decode("utf-8").replace("\n", "") for line in blockwords_file
    ]

    for thread in threads:
        comments = thread["comments"]
        comments_to_remove = []
        # Go through every comment chain one by one
        # If we detect a bad comment (bot or NSFW), mark it to be removed
        for comment in comments:
            # top level comments
            comments_to_remove += find_bot_nsfw(comment, blockwords)

        # print(f"Found {len(comments_to_remove)} bot/nsfw comments to be removed.")
        # for comment in comments_to_remove:
            # print(f"Comment found: {comment}")

        # Remove comments.
        new_thread = thread
        new_comments = []
        for comment in comments:
            # Remove top-level comments
            if comment['comment_id'] in comments_to_remove:
                new_thread['comments_length'] = new_thread['comments_length'] - (comment["chain_length"])

            # Remove non top-level comments
            else:
                new_comment = remove_replies(comment, comments_to_remove)
                if new_comment['chain_length'] != comment['chain_length']:
                    new_thread['comments_length'] = new_thread['comments_length'] - (comment['chain_length'] - new_comment['chain_length'])
                new_comments.append(new_comment)

        new_thread['num_comments'] = thread['num_comments'] - (len(comments_to_remove))
        new_thread['comments'] = new_comments

        # Check if more comments must be removed.
        if new_thread['num_comments']  > maxcomments:
            # Go through all comments, collect id + timestamp
            comment_utc_list = []
            for comment in new_thread['comments']:
                comment_utc_list += utc_comments(comment)
            
            # Sort according to timestamp
            # Cut off all oldest comments after maxcomments, add all of them to 'comments_to_remove'
            comment_utc_list.sort()
            comment_utc_to_remove = comment_utc_list[maxcomments:]
            comments_to_remove = []
            for comment in comment_utc_to_remove:
                comments_to_remove.append(comment[1])
        


            # Remove comments.
            new_new_comments = []
            for comment in new_comments:
                # Remove top-level comments
                if comment['comment_id'] in comments_to_remove:
                    new_thread['comments_length'] = new_thread['comments_length'] - (comment["chain_length"])
                    # Don't append this comment to our new comments at all

                # Remove non top-level comments
                else:
                    new_comment = remove_replies(comment, comments_to_remove)
                    if new_comment['chain_length'] != comment['chain_length']:
                        new_thread['comments_length'] = new_thread['comments_length'] - (comment['chain_length'] - new_comment['chain_length'])
                    new_new_comments.append(new_comment)

            new_thread['num_comments'] = thread['num_comments'] - (len(comments_to_remove))
            new_thread['comments'] = new_new_comments
        
        # Remove any URLs in the comments...
        for comment in new_thread['comments']:
            remove_urls(comment)
    
        new_threads.append(new_thread)
        # If still over max_comment, go through each comment and sort them according to timestamp
        # Then get rid of all newest comments until we only have 40

    new_threads_data['threads'] = new_threads
    with open(savefile, 'w') as f:
        json.dump(new_threads_data, f, indent=4)


# Returns a new comment with specified comment chains removed.
def remove_replies(comment, comments_to_remove):
    # If no replies, just return comment as-is
    new_comment = {
                        "comment_id": comment['comment_id'],
                        "body": comment['body'],
                        "score": comment['score'],
                        "author": comment['author'],
                        "created_utc": comment['created_utc'],
                        "permalink": comment['permalink'],
                        "is_submitter": comment['is_submitter'],
                        "chain_length": comment['chain_length'],
                        "replies": comment['replies'],
                    }
    if comment['replies'] == []:
        return new_comment
    else:
        new_replies = []
        for reply in comment['replies']:
            # Remove comment chain, update chain length
            if reply["comment_id"] in comments_to_remove:
                new_comment['chain_length'] = comment['chain_length'] - reply['chain_length']
            # Check further replies
            else: 
                new_reply = remove_replies(reply, comments_to_remove)
                if new_reply['chain_length'] != reply['chain_length']:
                    new_comment['chain_length'] = new_comment['chain_length'] - (reply['chain_length'] - new_reply['chain_length'])
                new_replies.append(new_reply)

        new_comment['replies'] = new_replies

        return new_comment

def utc_comments(comment):
    if comment['replies'] == []:
        return [[comment["created_utc"],comment['comment_id']]]
    else:
        replies = []
        for reply in comment["replies"]:
            replies += utc_comments(reply)
        comment_list = [[comment["created_utc"],comment['comment_id']]] + replies
        return comment_list

# Returns True if comment is likely a bot.
def is_bot_comment(comment):
    bot_words = ["auto", "bot"]
    for word in bot_words:
        if word in comment.lower():
            return True
    return False

# Returns True if comment is likely NSFW.
def is_nsfw_comment(comment, blockwords):
    for word in blockwords:
        if f" {word} " in f" {comment} ":
            return True
    return False

def remove_urls(comment):
    body = comment['body']
    url = url_finder.search(body)
    if url:
        new_body = url_finder.sub('[URL]', body)
        comment['body'] = new_body
    for reply in comment['replies']:
        remove_urls(reply)

# Returns a list of comment_ids that may be bots or NSFW.
def find_bot_nsfw(comment, blockwords):
    # Mark this comment chain to be remove if not or nsfw
    body = comment['body']
    if '[removed]' in body:
        return [comment['comment_id']]
    if '[deleted]' in body:
        return [comment['comment_id']]
    if (is_bot_comment(comment['author']) or is_nsfw_comment(body, blockwords)):
        return [comment['comment_id']]
    
    if comment['replies'] == []:
        return []

    # Check all remaining replies
    else:
        comments_to_remove = []
        for reply in comment['replies']:
            comments_to_remove += find_bot_nsfw(reply, blockwords)
    return comments_to_remove

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', type=str, help="Name of file to filter. Should be a thread json file.")
    parser.add_argument('-s', '--savefile', type=str, help="Name of file to save to. Should be a jsonfile")
    parser.add_argument('--filter', type=str, choices=['nsfw', 'adv', 'bot', 'all'], default='all')
    parser.add_argument('--maxcomments', type=int, default=50, help="Max number of comments to keep.")

    args = parser.parse_args()
    main(args)