import os
import json
import argparse

def main(args):
    filename = args.filename
    savefile = args.savefile
    assert args.prompt in ['clustergold', 'clusterpred']

    with open(filename, 'r') as f:
        data = json.load(f)
        threads = data['threads']

    cluster_id2pred_first_stage = {}
    if args.prompt == 'clusterpred':
        with open(args.pred_first_stage_tgt, 'r') as fp:
            pred_first_stage_tgt = fp.read().splitlines()
        for pred in pred_first_stage_tgt:
            words = pred.split()
            cluster_id2pred_first_stage[words[0]] = ' '.join(words[1:])

    processed_threads = []

    if args.use_image_caption:
        with open('/gallery_tate/keighley.overbay/thread-summarization/data/image_captions_blip2_caption_best.json', 'r') as f:
            imgcaps = json.load(f)

    for thread in threads:
        doc = ""
        doc_imgcap = ""
        # Add OP to document
        doc += "Original Post: "
        doc += thread['raw_caption'].replace('\n', ' ').replace('\r', ' ')

        if args.use_image_caption:
            sub_id = thread['submission_id']
            doc += f" Image: {imgcaps[sub_id]}."
        doc += args.separator

        if args.prompt == 'clustergold':
            csums = [x.replace('\n', ' ').replace('\r', ' ') for x in thread['csums']]
        elif args.prompt == 'clusterpred':
            csums = []
            for cluster_id, comments in thread['clusters_auto'].items():
                post_comments_id = '-'.join([thread['submission_id']]+thread['clusters_auto'][cluster_id]['comments'])
                csums.append(cluster_id2pred_first_stage[post_comments_id])
        else:
            raise ValueError
        processed_thread = {}
        processed_thread['document'] = f"{doc} {' '.join(csums)}"
        processed_thread['summary'] = thread["edited_sum"].replace('\n', ' ').replace('\r', ' ')
        processed_thread['id'] = f"{thread['submission_id']}"
        processed_threads.append(processed_thread)

    with open(savefile, 'w') as f:
        json.dump({"threads": processed_threads}, f, indent=4)

def comment_to_string(comment, sep):
    string = f"{comment['author_anon']}: {comment['body']}".replace('\n',' ').replace('\r', ' ')
    if comment['replies'] == []:
        return string
    for reply in comment['replies']:
        string += sep
        string += comment_to_string(reply, sep)
    return string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prompt', type=str, choices=['imgcap', 'glip', 'oponly', 'oponlyimgcap', 'clustergold', 'clusterpred', 'none'], default='none', help="Type of prompt to add.")
    parser.add_argument('-sep', '--separator', type=str, default="", help="The separator to go between each comment. Used for HT5 where the separator is |||.")
    parser.add_argument('-f', '--filename', type=str, help="Name of file to filter. Should be a thread json file.")
    parser.add_argument('-s', '--savefile', type=str, help="Name of file to save to. Should be a jsonfile")
    parser.add_argument('-predfirst', '--pred_first_stage_tgt', type=str)
    parser.add_argument('-imgcap', "--use_image_caption", action='store_true')

    args = parser.parse_args()
    main(args)
