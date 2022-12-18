import os
import json

# Read in all subreddits
subreddits = list()
# with open("../subreddits.txt", 'r') as f:
#     subreddits = [line.rstrip() for line in f]
subreddits.append('r/outfits')
subreddits.append('r/fashionadvice')
subreddits_to_read = subreddits
total_counts = {}

# Execute download command for each
for sub in subreddits_to_read:
    for month in ['09']:
    # for month in ['01', '02','03','04','05','06','07','08']: #,'10','11','12']:
    # for month in ['01', '02','03','04','05','06','07','08','09','10','11','12']:
        for year in ['2022']:
        # for year in ['2019', '2020', '2021']: #  ['2022']:
            filename_under10 = f"datasets/redcaps/annotations_5-9comments/{sub[2:]}_{year}-{month}.json"
            filename_over10 = f"datasets/redcaps/annotations/{sub[2:]}_{year}-{month}.json"
            filename_combined = f"datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}.json"
            # filename_under10 = f"datasets/redcaps/annotations_5-9comments/{sub[2:]}_{year}-{month}_filtered.json"
            # filename_over10 = f"datasets/redcaps/annotations/{sub[2:]}_{year}-{month}_filtered.json"
            # filename_combined = f"datasets/redcaps/annotations_5+comments/{sub[2:]}_{year}-{month}_filtered.json"
            if os.path.exists(filename_combined):
                print("Uh oh - already file in 5+ comments...check your stuff Keighley wtf u doing")
                continue
            if os.path.exists(filename_under10):
                if os.path.exists(filename_over10):
                    with open(filename_under10, 'r') as f:
                        data_under10 = json.load(f)
                    with open(filename_over10, 'r') as f:
                        data_over10 = json.load(f)
                    new_threads = data_under10['threads'] + data_over10['threads']
                    new_data = {'info': data_under10['info'], 'threads': new_threads}
                    with open(filename_combined, 'w') as f:
                        json.dump(new_data, f, indent=4)

