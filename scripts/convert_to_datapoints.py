import os
import json
import argparse

def main(args):
    filename = args.filename
    savefile = args.savefile

    with open(filename, 'r') as f:
        data = json.load(f)
        threads = data['threads']
    
    for thread in threads:
        doc = ""
        doc_imgcap = ""
        # Add OP to document
        doc += "Original Post: "
        doc += thread['raw_caption']

        doc_imgcap += "Original Post: "
        doc_imgcap += thread['raw_caption']

        comment_number = 1
        for tl_comment in thread['comments']:
            cts = comment_to_string(tl_comment, str(comment_number))
            doc += "\n"
            doc += cts
            doc_imgcap += "\n"
            doc_imgcap += cts
            comment_number += 1
        
        thread['document'] = doc

        with open(savefile, 'w') as f:
            json.dump(data, f, indent=4)


def comment_to_string(comment, comment_number):
    string = f"Comment {str(comment_number)}: {comment['body']}"
    if comment['replies'] == []:
        return string
    new_comment_number = 1
    for reply in comment['replies']:
        string += "\n"
        string += comment_to_string(reply, (comment_number + '.' + str(new_comment_number)))
        new_comment_number += 1
    return string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', type=str, help="Name of file to filter. Should be a thread json file.")
    parser.add_argument('-s', '--savefile', type=str, help="Name of file to save to. Should be a jsonfile")

    args = parser.parse_args()
    main(args)
