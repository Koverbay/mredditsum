import json
import argparse


def main(args):
    filepath = args.f
    with open(filepath, 'r') as f:
        data = json.load(f)
    threads = data['threads']
    
    for thread in threads:
        names = {}
        names[thread['author']] = "OP"
        thread['author_anon'] = "OP"

        for comment in thread['comments']:
            replace_names(comment, names)
            

    with open(args.s, 'w') as f:
        json.dump(data, f, indent=4)

def replace_names(comment, names):
    if not (comment['author'] in names):
        names[comment['author']] = f"User {len(names)}"
    comment['author_anon'] = names[comment['author']]
    for reply in comment['replies']:
        replace_names(reply, names)
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True, type=str)
    parser.add_argument('-s', required=True, type=str)
    args = parser.parse_args()
    main(args)
