import os

# Read in all subreddits
subreddits = list()
# with open("../new_subreddits.txt", 'r') as f:
#     subreddits = [line.rstrip() for line in f]
subreddits.append('r/outfits')
subreddits_to_read = subreddits

# Execute download command for each
for sub in subreddits_to_read:
    for year in ['2015', '2016', '2017', '2018']:
        for month in ['01', '02','03','04','05','06','07','08','09', '10','11','12']:
    
    # for year in ['2022']:
    #     for month in ['01', '02','03','04','05','06','07','08','09', '10']:
    # for year in ['2019', '2020', '2021']: # #
    #     for month in ['01', '02','03','04','05','06','07','08','09', '10','11','12']:
        # for month in ['09', '10']:

    #     for year in ['2019', '2020', '2021']:
            os.system(f"redcaps download-anns --subreddit {sub[2:]} -m {year}-{month} -o ../redcaps-downloader/datasets/annotations_5+comments -c '../redcaps-downloader/credentials.json'")
            # os.system(f"redcaps download-anns --subreddit {sub[2:]} -m {year}-{month} -o ./datasets/redcaps/annotations -c '././redcaps-downloader/credentials.json'")
    print(f"Finished downloading from subreddit {sub}!")