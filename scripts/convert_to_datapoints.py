import os
import json
import pickle
import argparse

def main(args):
    filename = args.filename
    savefile = args.savefile

    with open(filename, 'r') as f:
        data = json.load(f)
        threads = data['threads']

    if args.prompt == 'clusterpred':
        with open(args.predclusterids, 'rb') as fp:
            pred_clusters = pickle.load(fp)

    processed_threads = []

    if (args.prompt == 'imgcap') or (args.prompt == 'oponlyimgcap'):
        with open('/gallery_tate/keighley.overbay/thread-summarization/data/image_captions_blip2_caption_best.json', 'r') as f:
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
            if args.prompt in ['clustergold', 'clusterpred']:
                comment_id2comments = {}
                for tl_comment in thread['comments']:
                    comment_id2comments[tl_comment['comment_id']] = comment_to_string(tl_comment, args.separator)
                cluster_id2comments = {}
                cluster_comment_id2cluster_id = {}
                cnt_tot, cnt_selected = 0, 0
                clusters = {}
                for cluster_id, comments in thread['clusters_auto'].items():
                    cluster_comments = [comment_id2comments[x] for x in comments['comments']]
                    cluster_id2comments[cluster_id] = ' '.join(cluster_comments)
                    cluster_comment_id2cluster_id['-'.join(comments['comments'])] = cluster_id
                    if args.prompt == 'clustergold':
                        if cluster_id in thread['csums_with_ids'].keys():
                            doc += " "
                            doc +=  cluster_id2comments[cluster_id]
                            doc += args.separator
                    elif args.prompt == 'clusterpred':
                        cnt_tot += 1
                        cluster_comment_id = '-'.join(comments['comments'])
                        if cluster_comment_id in pred_clusters.keys():
                            clusters[cluster_comment_id] = pred_clusters[cluster_comment_id]
                            ### cnt_selected += 1
                            ### doc += " "
                            ### doc +=  cluster_id2comments[cluster_id]
                            ### doc += args.separator
                    else:
                        raise ValueError

                if args.prompt == 'clusterpred':
                    if len(clusters) <= 5:
                        for cluster_comment_id in clusters.keys():
                            cnt_selected += 1
                            pred_cluster_id = cluster_comment_id2cluster_id[cluster_comment_id]
                            doc += " "
                            doc += cluster_id2comments[pred_cluster_id]
                            doc += args.separator
                    else:
                        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['prob'], reverse=True)
                        prob_threshold = sorted_clusters[4][1]['prob']
                        for cluster_comment_id in clusters.keys():
                            if clusters[cluster_comment_id]['prob'] >= prob_threshold:
                                cnt_selected += 1
                                pred_cluster_id = cluster_comment_id2cluster_id[cluster_comment_id]
                                doc += " "
                                doc += cluster_id2comments[pred_cluster_id]
                                doc += args.separator
                        assert cnt_selected <= 5
                print(f'selected: {cnt_selected} among total: {cnt_tot}')
            else:
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

    parser.add_argument('-p', '--prompt', type=str, choices=['imgcap', 'glip', 'oponly', 'oponlyimgcap', 'clustergold', 'clusterpred', 'none'], default='none', help="Type of prompt to add.")
    parser.add_argument('-sep', '--separator', type=str, default="", help="The separator to go between each comment. Used for HT5 where the separator is |||.")
    parser.add_argument('-f', '--filename', type=str, help="Name of file to filter. Should be a thread json file.")
    parser.add_argument('-s', '--savefile', type=str, help="Name of file to save to. Should be a jsonfile")
    parser.add_argument('-pci', '--predclusterids', type=str, help="Name of file to filter by cluster ids.")

    args = parser.parse_args()
    main(args)
