import pickle
import json

# data = [json.loads(row) for row in ]
# print(len(data))

def comment_to_string(comment):
    if comment['author_anon'] == "OP":
        author = "OP"
    else:
        author = f"Commenter {comment['author_anon'][-1]}"

    string = f"{author}: {comment['body']}"
    for reply in comment['replies']:
        string += "\n"
        string += comment_to_string(reply)
    return string

f = open('/gallery_tate/keighley.overbay/thread-summarization/thread_data/full_dataset/full_edited.json')
data = json.load(f)

data_dict = {}
for thread in data['threads']:
    sub_id = thread["submission_id"]
    caption = "OP: " + thread["raw_caption"] + " [SEP] "
    comments = ""
    for tl_com in thread["comments"]:
        comments += "\n"
        cts = comment_to_string(tl_com)
        comments += cts
        
        # com_id = coms["comment_id"]

    data_dict[sub_id] = {
                "sub_id": sub_id,
                "caption": caption,
                "cap_com": caption + comments,
                "edited_summaries": thread['edited_sum']
            }

json.dump(data_dict, open("thread_caption_comments.json","w"), indent=4)

