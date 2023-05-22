import os
import json
import argparse

def main(args):
    filename = args.filename
    savefile = args.savefile

    with open(filename, 'r') as f:
        data = json.load(f)
        threads = data['threads']
    
    processed_threads = []

    if (args.prompt == 'imgcap') or (args.prompt == 'oponlyimgcap'):
        with open('../data/image_captions_blip2_caption_best.json', 'r') as f:
            imgcaps = json.load(f)

    for thread in threads:
        doc = ""
        doc_imgcap = ""
        # Add OP to document
        doc += "Original Post: "
        doc += thread['raw_caption'].replace('\n', ' ').replace('\r', ' ')        
        
        if (args.prompt == 'imgcap') or (args.prompt == 'oponlyimgcap') :
            sub_id = thread['submission_id']
            doc += f" Image: {imgcaps[sub_id]}."
        doc += args.separator
        
        if (args.prompt != 'oponly') and (args.prompt != 'oponlyimgcap'):
            for tl_comment in thread['comments']:
                cts = comment_to_string(tl_comment, args.separator) #, str(comment_number))
                doc += " "
                doc += cts
                doc += args.separator
        processed_thread = {}
        processed_thread['document'] = doc
        if (args.prompt == 'oponly') or (args.prompt == 'oponlyimgcap'):
            processed_thread['summary'] = thread["opsum"].replace('\n',' ').replace('\r', ' ')
        else:
            processed_thread['summary'] = thread["edited_sum"].replace('\n',' ').replace('\r', ' ')
        processed_thread['id'] = thread["submission_id"]
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

    parser.add_argument('-p', '--prompt', type=str, choices=['imgcap', 'glip', 'oponly', 'oponlyimgcap', 'none'], default='none', help="Type of prompt to add.")
    parser.add_argument('-sep', '--separator', type=str, default="", help="The separator to go between each comment. Used for HT5 where the separator is |||.")
    parser.add_argument('-f', '--filename', type=str, help="Name of file to filter. Should be a thread json file.")
    parser.add_argument('-s', '--savefile', type=str, help="Name of file to save to. Should be a jsonfile")

    args = parser.parse_args()
    main(args)
