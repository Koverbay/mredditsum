import json
import argparse
import os
import shutil


def main(args):

    interior_path = "../thread_data/threads_to_annotate/comments_10+/json/interior_all.json"
    fashion_path = "../thread_data/threads_to_annotate/comments_10+/json/fashion_all.json"

    with open(interior_path, 'r') as f:
        interior_data = json.load(f)
        interior_threads = interior_data['threads']
    with open(fashion_path, 'r') as f:
        fashion_data = json.load(f)
        fashion_threads = fashion_data['threads']


    # Save threads in batches with a 3:1 ratio.
    batch_no = 0
    # no_threads = 300
    no_fashion_threads = 75
    no_interior_threads = 225
    while (interior_threads != [] or fashion_threads != []):
        if len(interior_threads) >= no_interior_threads:
            batch_interior_threads = interior_threads[:no_interior_threads]
            interior_threads = interior_threads[no_interior_threads:]
        else:
            print(f"Not enough threads for interior. Saving remaining {len(interior_threads)} instead.")
            batch_interior_threads = interior_threads
            interior_threads = []
            
        if len(fashion_threads) >= no_fashion_threads:
            batch_fashion_threads = fashion_threads[:no_fashion_threads]
            fashion_threads = fashion_threads[no_fashion_threads:]
        else:
            print(f"Not enough threads for fashion. Saving remaining {len(fashion_threads)} instead.")
            batch_fashion_threads = fashion_threads
            fashion_threads = []
        batch_threads = batch_interior_threads + batch_fashion_threads
        no_threads = len(batch_threads)
        batch_path = f"batch{batch_no}_{no_threads}/batch{batch_no}_{no_threads}.json"

        print(f"Saving {no_threads} to file {batch_path}...")
        with open(args.savepath + batch_path, 'w') as f:
            json.dump({"threads": batch_threads},f,indent=4)

        print("Copying images to new directory...")
        for thread in batch_threads:
            old_image_fp = f"../thread_data/threads_to_annotate/comments_10+/images/{thread['submission_id']}.jpg"
            new_image_fp = f"../thread_data/batch{batch_no}_{no_threads}/images/{thread['submission_id']}.jpg"
            if os.path.exists(old_image_fp):
                shutil.copy(old_image_fp, new_image_fp)
            else:
                print(f"Oops! Image not found for id {thread['submission_id']}")
        batch_no += 1
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--savepath", default="../thread_data/")
    args = parser.parse_args()
    main(args)