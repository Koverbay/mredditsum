import os
import json
import glob

# Read in all subreddits
subreddits = list()
# with open("../subreddits.txt", 'r') as f:
#     subreddits = [line.rstrip() for line in f]
subreddits.append('r/outfits')
total_counts = {}


# Execute download command for each
for sub in subreddits:
    for year in ['2019', '2020', '2021', '2022']:
        for month in ['01', '02','03','04','05','06','07','08','09','10','11','12']:
            filepath = f"datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json"
            if os.path.exists(filepath):
                os.system(f"python3 blur_faces.py -f {filepath}")
            else:
                print(f"Oops! No file {filepath}")
        # print(f"Finished counting subreddit {sub}! Total threads: {total_threads}")

    blurred_count = len(glob.glob1(f"datasets/redcaps/images/{sub[2:]}", '*_blurred*'))
    total_counts[sub] = blurred_count

print(f"Finished filtering all subreddits. Counts:")
for sub in subreddits:
    print(f"Total faces blurred from subreddit {sub}: {total_counts[sub]}")
