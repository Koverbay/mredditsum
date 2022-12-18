import os
import json

# Read in all subreddits
subreddits = list()
# with open("../new_subreddits.txt", 'r') as f:
#     subreddits = [line.rstrip() for line in f]
subreddits.append('r/fashionadvice')
subreddits_to_read = subreddits
total_counts_under10 = {}
total_counts_over10 = {}


# Execute download command for each
for sub in subreddits_to_read:
    total_threads_under10 = 0
    total_threads_over10 = 0
    for year in ['2022']:
    # for year in  ['2019', '2020', '2021', '2022']:
    # for year in ['2015', '2016', '2017', '2018']:
        for month in ['11']:
        # for month in ['01', '02','03','04','05','06','07','08','10','11','12']:
        # for month in ['01', '02','03','04','05','06','07','08','09','10','11','12']:
            filepath = f"../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}.json"
            if os.path.exists(filepath) and not (os.path.exists(f"../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json")):
                os.system(f"python3 filter_threads.py -f ../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}.json -s ../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json")
                os.system(f"redcaps download-imgs --annotations ../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json --resize 512 -j 4 -o ../redcaps-downloader/datasets/images --update-annotations")
                os.system(f"python3 filter_comments.py -f ../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json -s ../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json")
                
            if os.path.exists(f"../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json"):
                with open(f"../redcaps-downloader/datasets/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json", 'r') as f:
                    data = json.load(f)
                    threads = data['threads']
                for thread in threads:
                    if thread['num_comments'] >= 10:
                        total_threads_over10 += 1
                    else:
                        total_threads_under10 += 1
            else:
                print(f"Oops! No file {filepath}")
        total_counts_over10[sub] = total_threads_over10
        total_counts_under10[sub] = total_threads_under10

print(f"Finished filtering all subreddits. Counts:")
for sub in subreddits_to_read:
    print(f"Total threads with 10+ comments in {sub}: {total_counts_over10[sub]}")
    print(f"Total threads with 5-9 comments in {sub}: {total_counts_under10[sub]}")