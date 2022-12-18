import os
import json

# Read in all subreddits
subreddits = list()
with open("../subreddits.txt", 'r') as f:
    subreddits = [line.rstrip() for line in f]
subreddits_to_read = subreddits
total_counts_under10 = {}
total_counts_over10 = {}


# Execute download command for each
for sub in subreddits_to_read:
    total_threads_under10 = 0
    total_threads_over10 = 0
    # for month in ['01', '02','03','04','05','06','07','08']: #,'10','11','12']:
    for month in ['01', '02','03','04','05','06','07','08','09','10','11','12']:
        # for year in ['2022']:
        # for year in ['2019', '2020', '2021']: #  ['2022']:
        for year in ['2019', '2020', '2021', '2022']:
            # os.system(f"python3 filter_threads.py -f datasets/redcaps/annotations_5-9comments/{sub[2:]}_{year}-{month}.json -s datasets/redcaps/annotations_5-9comments/{sub[2:]}_{year}-{month}_filtered.json")
            # os.system(f"python3 filter_comments.py -f datasets/redcaps/annotations_5-9comments/{sub[2:]}_{year}-{month}_filtered.json -s datasets/redcaps/annotations_5-9comments/{sub[2:]}_{year}-{month}_filtered.json")
            # os.system(f"redcaps download-imgs --annotations ./datasets/redcaps/annotations_5-9comments/{sub[2:]}_{year}-{month}_filtered.json --resize 512 -j 4 -o ./datasets/redcaps/images --update-annotations")
            # with open(f"datasets/redcaps/annotations_5-9comments/{sub[2:]}_{year}-{month}_filtered.json", 'r') as f:
            # os.system(f"python3 filter_threads.py -f datasets/redcaps/annotations/{sub[2:]}_{year}-{month}.json -s datasets/redcaps/annotations/{sub[2:]}_{year}-{month}_filtered.json")
            # os.system(f"python3 filter_comments.py -f datasets/redcaps/annotations/{sub[2:]}_{year}-{month}_filtered.json -s datasets/redcaps/annotations/{sub[2:]}_{year}-{month}_filtered.json")
            # os.system(f"redcaps download-imgs --annotations ./datasets/redcaps/annotations/{sub[2:]}_{year}-{month}_filtered.json --resize 512 -j 4 -o ./datasets/redcaps/images --update-annotations")
            # with open(f"datasets/redcaps/annotations/{sub[2:]}_{year}-{month}_filtered.json", 'r') as f:
            filepath = f"datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json"
            if os.path.exists(filepath):
                # os.system(f"python3 filter_threads.py -f datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}.json -s datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json")
                # os.system(f"python3 filter_comments.py -f datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json -s datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json")
                # os.system(f"redcaps download-imgs --annotations ./datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json --resize 512 -j 4 -o ./datasets/redcaps/images --update-annotations")
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    threads = data['threads']
                for thread in threads:
                    if thread['num_comments'] >= 10:
                        total_threads_over10 += 1
                    else:
                        total_threads_under10 += 1
                # total_threads += len(threads)
            else:
                print(f"Oops! No file {filepath}")
        # print(f"Finished counting subreddit {sub}! Total threads: {total_threads}")
        total_counts_over10[sub] = total_threads_over10
        total_counts_under10[sub] = total_threads_under10

print(f"Finished counting all subreddits. Counts:")
for sub in subreddits_to_read:
    print(f"Total threads with 10+ comments in {sub}: {total_counts_over10[sub]}")
    print(f"Total threads with 5-9 comments in {sub}: {total_counts_under10[sub]}")