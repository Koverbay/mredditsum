import json
import argparse
import csv
import ftfy
import re
import os

def main(args):

    write_data = []
    fpath = args.filepath
    
    if not os.path.exists(fpath):
        print(f"Oops! File not found: '{fpath}'")
        exit()
    with open(fpath, 'r') as f:
        data = json.load(f)
    threads = data['threads']

    task = args.task
    savefile = args.filepath[:(len(args.filepath)-5)] + f"_{task}.csv"
    # Prepare data for "OP Summarization Task."
    if task == "opsum":

        for thread in threads:
            write_data.append((thread['submission_id'], thread['url'], thread['caption']))

        print(f"Writing data to file {savefile}...")
        with open(savefile, 'w', newline='') as csvfile:
            fieldnames = [ 'id', 'image', 'post']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in write_data:
                writer.writerow({'id': item[0], 'image': item[1], 'post': item[2]})

        print("Finished writing!")

    # Prepare data for "Cluster Summarization Task".
    if task == "csum":

        for thread in threads:
            clusters_html_writable = []
            # Get the list of clusters
            clusters = thread['clusters_auto']
            clusters_list = []
            for cluster in clusters:
                clusters_list.append((clusters[cluster]['score'], clusters[cluster]['comments']))
            # Top scoring clusters first
            clusters_list.sort(reverse=True)

            for cluster in clusters_list[:6]: # Top 6 clusters only
                cluster_html_writable = []
                for top_level_comment_id in cluster[1]:
                    # Get the actual comment object with replies, etc
                    full_comment = {}
                    for top_level_comment in thread['comments']:
                        if top_level_comment_id == top_level_comment['comment_id']:
                            full_comment = top_level_comment
                            break
                                        
                    comment_html_writable = comment_to_html(full_comment, 0)
                    cluster_html_writable.append(comment_html_writable)
                clusters_html_writable.append (cluster_html_writable)
                print(cluster_html_writable)
                
                #[[<HTML STUFF 1>, <HTML STUFF 2>],[],...]

            write_data.append((thread['submission_id'], thread['url'], thread['caption'], clusters_html_writable))
        
        print(f"Writing data to file {savefile}...")
        with open(savefile, 'w', newline='') as csvfile:
            fieldnames = ['id', 'image', 'post', 'comments']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in write_data:
                writer.writerow({'id': item[0], 'image': item[1], 'post': item[2], 'comments': item[3]})

        print("Finished writing!")

    # Prepare data for "Final Summarization Task".
    if task == "fsum":
        
        for thread in threads:
            write_data.append((thread['submission_id'], thread['url'], thread['caption'], thread['unedited_sum']))

        print(f"Writing data to file {savefile}...")
        with open(savefile, 'w', newline='') as csvfile:
            fieldnames = [ 'id', 'image', 'post', 'origsum']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in write_data:
                writer.writerow({'id': item[0], 'image': item[1], 'post': item[2], 'origsum': item[3]})

        print("Finished writing!")


            
def comment_to_html(comment, depth):
    # Add tabs according to depth:
    # html = ""
    # html += '&emsp; '*depth
    # Bolded username
    html = ""
    html += '<b>' + comment['author_anon'] + ':</b> '
    # Sanitized body
    html += sanitize_comment(comment['body'])
    # Newline
    html += '<br/>'
    if comment["replies"] != []:
        html += '<ul>'
        for reply in comment["replies"]:
            html += '<li>'
            html += comment_to_html(reply, depth+1)
            html += '</li>'
        html += '</ul>'
    return html

def sanitize_comment(comment):

        regex_candidates = [r"[\[\(].*?[\]\)]", r"\s*\d+\s*[x×\*\,]\s*\d+\s*"]

        # First remove all accents and widetexts from caption.
        comment = ftfy.fix_text(comment, normalization="NFKD").lower()

        # Remove all above regex candidates.
        for regexc in regex_candidates:
            comment = re.sub(regexc, "", comment, flags=re.IGNORECASE)

            # Remove multiple whitespaces, and leading or trailing whitespaces.
            # We have to do it every time we remove a regex candidate as it may
            # combine surrounding whitespaces.
            comment = re.sub(r"\s+", " ", comment).strip()

        # In this end, replace all usernames with `<usr>` token.
        comment = re.sub(r"\@[_\d\w\.]+", "<usr>", comment, flags=re.IGNORECASE)

        # Remove all emojis and non-latin characters.
        comment = comment.encode("ascii", "ignore").decode("utf-8")
        return comment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True)
    # parser.add_argument("-s", "--savefile", required=True)
    parser.add_argument("-t", "--task", default="opsum", choices=["opsum",  "csum", "fsum"])
    args = parser.parse_args()
    main(args)