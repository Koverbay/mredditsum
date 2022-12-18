from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import argparse
import json
import colorful as cf

def main(args):
    filepath = args.f
    with open(filepath, 'r') as f:
        data = json.load(f)
    threads = data['threads']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    display = True
    
    for thread in threads:
        clusters = {}
        labeled_comments = {}
        ids = []
        sentences = []
        scores = []
        # Get all top-level comments
        for comment in thread['comments']:
            ids.append(comment['comment_id'])
            sentences.append(comment['body'])
            scores.append(comment['score'])

        embeddings = model.encode(sentences)
        clustering = AgglomerativeClustering(affinity='cosine', n_clusters=None, linkage='average', distance_threshold=.50).fit(embeddings)
        index = 0
        # Make a dictionary with cluster ids, as well as one that also includes sentences + scores
        for label in clustering.labels_:
            if not (label in labeled_comments):
                clusters[int(label)] = {'comments': [ids[index]]}
                labeled_comments[int(label)] = {'comments': [(ids[index], sentences[index], scores[index])]}
            else:
                clusters[label]['comments'].append(ids[index])
                labeled_comments[label]['comments'].append((ids[index], sentences[index], scores[index]))
            index += 1
        

        # Calculate the score for each cluster
        for cluster in labeled_comments:
            cluster_score = 0
            for comment in labeled_comments[cluster]['comments']:
                cluster_score += comment[2]
            clusters[cluster]['score'] = cluster_score

        # if display:
        #     print(cf.purple("Displaying OP and comment clusters..."))

        #     print(cf.yellow(f"Original Post: "),cf.orange(thread['raw_caption']))
        #     for cluster in labeled_comments:
        #         for sentence in labeled_comments[cluster]:
        #             print(cf.yellow(f"Comment from cluster {cluster}: "),cf.green(sentence[1]))


        #     print(cf.purple("Enter 'c' to continue clustering without displaying"))
        #     response = input()
        #     if response == 'c':
        #         display = False

        thread['clusters_auto'] = clusters

    with open(args.s, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True, type=str)
    parser.add_argument('-s', required=True, type=str)
    args = parser.parse_args()
    main(args)