import os

# Read in all subreddits
subreddits = list()
with open("../new_subreddits.txt", 'r') as f:
    subreddits = [line.rstrip() for line in f]

# Choose which subreddits to read
subreddits_to_read = subreddits[14:]

# Execute download command for each
for sub in subreddits_to_read:
    os.system(f"redcaps download-imgs --annotations ./datasets/redcaps/annotations/{sub[2:]}_2022-07.json --resize 512 -j 4 -o ./datasets/redcaps/images --update-annotations")
    print(f"Finished downloading from subreddit {sub}!")