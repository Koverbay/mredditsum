import argparse
import json
import re

# Keywords for threads asking for advice or opinions
keywords = ['advice', 'opinion', 'opinions', 'recommend', 'recommendations', 'thought', 'thoughts', 'think', 'what', 'how', 'should', 'would', 'idea', 'ideas', 'suggest', 'suggestions', 'tip', 'tips', 'help', 'feedback', 'input']
 
def main(args):
    filename = args.filename
    savefile = args.savefile

    with open(filename, 'r') as f:
        threads_data = json.load(f)
        threads = threads_data['threads']
    new_threads_data = threads_data
    new_threads = []

    for thread in threads:
        # Remove threads that do not contain one of the keywords in post caption
        caption = thread['caption']
        added = False
        for keyword in keywords:
            clean_caption = re.sub(r"[,.!?]", "", caption)
            if keyword in clean_caption.split():
                # print(f"found keyword in caption ({caption})...")
                new_threads.append(thread)
                added = True
                break
    # print(f"Removing post ({caption})...")
        # print(len(new_threads))
        if added == False:
            print(f"Removing post ({caption})...")

    new_threads_data['threads'] = new_threads
    with open(savefile, 'w') as f:
        json.dump(new_threads_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', type=str, help="Name of file to filter. Should be a thread json file.")
    parser.add_argument('-s', '--savefile', type=str, help="Name of file to save to. Should be a jsonfile")

    args = parser.parse_args()
    main(args)