import pickle
import json

# data = [json.loads(row) for row in ]
# print(len(data))

f = open('/gallery_tate/keighley.overbay/thread-summarization/thread_data/full_dataset/full_edited.json')
data = json.load(f)

data_dict = {}
for thread in data['threads']:
    sub_id = thread["submission_id"]
    caption = thread["caption"] + " [SEP] "
    comments = []
    for coms in thread["comments"]:
        com_id = coms["comment_id"]
        comments.append(coms['body'])
        for replies in coms["replies"]:
            comments.append(replies["body"])

    str_comms = " ".join(comments)
    data_dict[sub_id] = {
                "sub_id": sub_id,
                "caption": caption,
                "cap_com": caption + str_comms,
                "edited_summaries": thread['edited_sum']
            }

json.dump(data_dict, open("thread_caption_comments.json","w"))