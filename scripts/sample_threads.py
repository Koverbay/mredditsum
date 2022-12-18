"""
Samples threads by subreddit, and saves them in
a single file to be annotated.
""" 

import argparse
import json
import random
import shutil
import os

def main(args):

    new_data = {}
    new_data['threads'] = []

    filepath = f"/gallery_tate/keighley.overbay/thread-summarization/redcaps-downloader/datasets/annotations_5+comments/{args.sub}_{args.month}_filtered.json"
    with open(filepath, 'r') as f:
        data = json.load(f)
    threads = data['threads']
    if args.commentsoverten:
        print('over10')
        tenthreads = []
        for thread in threads:
            if thread['num_comments'] >= 10:
                tenthreads.append(thread)
        threads = tenthreads     
    else:
        print('undert10')
        undertenthreads = []
        for thread in threads:
            if thread['num_comments'] < 10:
                undertenthreads.append(thread)
        threads = undertenthreads   
        
    num_threads = len(threads)
    number = args.number
    month = args.month

    if num_threads >= number:
        print(f"Saving {number} threads from {month}...")
        for i in range(number):
            thread = threads[i]
            thread['month'] = month
            new_data['threads'].append(thread)
    
    while num_threads < number:
        print(f"{number} threads requested, but only {num_threads} found in month {month}. Check following months? y/n")
        x = input()

        print(f"Saving {num_threads} threads from {month}...")
        # Get threads from this month
        for i in range(num_threads):
            thread = threads[i]
            thread['month'] = month
            new_data['threads'].append(thread)

        if x == "n":
            #     # Save the selected threads
            # print("Saving json data...")
            # savepath = args.savefolder + f"/json/{args.sub}.json"
            # with open(savepath, 'w') as f:
            #     json.dump(new_data,f,indent=4)

            # # Copy the images to the directory as well
            # print("Copying images to new directory...")
            # for thread in new_data['threads']:
            #     old_image_fp = f"/gallery_tate/keighley.overbay/thread-summarization/redcaps-downloader/datasets/images/{args.sub}/{thread['submission_id']}.jpg"
            #     if os.path.exists(old_image_fp):
            #         new_image_fp = args.savefolder + f"/images/{thread['submission_id']}.jpg"
            #         shutil.copy(old_image_fp, new_image_fp)
            #     else:
            #         print(f"Oops! Image not found for id {thread['submission_id']}")
            break

        if x == "y":

            # print(f"Saving {num_threads} threads from {month}...")
            # # Get threads from this month
            # for i in range(num_threads):
            #     thread = threads[i]
            #     thread['month'] = month
            #     new_data['threads'].append(thread)

            number = (number - num_threads)
            m = int(month[len(month)-2:])
            if m < 12:
                if m < 9:
                    month = month[:len(month)-2] + '0' + str(m+1)
                else:
                    month = month[:len(month)-2] + str(m+1)
            else:
                month = str(int(month[:4])+1)+"-01"
            
            filepath = f"/gallery_tate/keighley.overbay/thread-summarization/redcaps-downloader/datasets/annotations_5+comments/{args.sub}_{month}_filtered.json"
            if not os.path.exists(filepath):
                print(f"Oops! No more data for subreddit {args.sub}. Filepath: {filepath} Saving and exiting...")
                break
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            threads = data['threads']
            if args.commentsoverten:
                print('over10')
                tenthreads = []
                for thread in threads:
                    if thread['num_comments'] >= 10:
                        tenthreads.append(thread)
                threads = tenthreads  
            else:
                print('undert10')
                undertenthreads = []
                for thread in threads:
                    if thread['num_comments'] < 10:
                        undertenthreads.append(thread)
                threads = undertenthreads
            num_threads = len(threads)
            print(f"Found {num_threads} from {month}...")
    


        


    # Save the selected threads
    print("Saving json data...")
    savepath = args.savefolder + f"/json/{args.sub}.json"
    with open(savepath, 'w') as f:
        json.dump(new_data,f,indent=4)

    # Copy the images to the directory as well
    print("Copying images to new directory...")
    for thread in new_data['threads']:
        old_image_fp = f"/gallery_tate/keighley.overbay/thread-summarization/redcaps-downloader/datasets/images/{args.sub}/{thread['submission_id']}.jpg"
        if os.path.exists(old_image_fp):
            new_image_fp = args.savefolder + f"/images/{thread['submission_id']}.jpg"
            shutil.copy(old_image_fp, new_image_fp)
        else:
            print(f"Oops! Image not found for id {thread['submission_id']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sub", required=True)
    parser.add_argument("-n", "--number", required=True, type=int)
    parser.add_argument("-m", "--month", required=True)
    parser.add_argument("-sf", "--savefolder", required=True)
    parser.add_argument("-c", "--commentsoverten", action="store_true")
    args = parser.parse_args()
    main(args)