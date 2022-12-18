import json
import argparse


def main(args):

    # thread_files = ["designmyroom_2019_0_clean.json",
    #                 "designmyroom_2019_1_clean.json",
    #                 "designmyroom_2020_0_clean.json",
    #                 "designmyroom_2020_1_clean.json",
    #                 "designmyroom_2020_2_clean.json",
    #                 "designmyroom_2021_0_clean.json",
    #                 "designmyroom_2021_1_clean.json",
    #                 "designmyroom_2021_2_clean.json",
    #                 "designmyroom_2022_0_clean.json",
    #                 "designmyroom_2022_1_clean.json",
    #                 "femalelivingspace_2019_clean.json",
    #                 "femalelivingspace_2020_clean.json",
    #                 "femalelivingspace_2021_clean.json",
    #                 "femalelivingspace_2022_clean.json",
    #                 "malelivingspace_2019_01_clean.json",
    #                 "malelivingspace_2019_02_clean.json",
    #                 "malelivingspace_2019_03_clean.json",
    #                 "malelivingspace_2020_01_clean.json",
    #                 "malelivingspace_2020_02_clean.json",
    #                 "malelivingspace_2020_03_clean.json",
    #                 "malelivingspace_2021_01_clean.json",
    #                 "malelivingspace_2021_02_clean.json",
    #                 "malelivingspace_2021_03_clean.json",
    #                 "malelivingspace_2022_01_clean.json",
    #                 "malelivingspace_2022_02_clean.json",
    #                 "malelivingspace_2022_03_clean.json",
    #                 ]
    thread_files = [
                    "fashionadvice_2019_clean.json",
                    "fashionadvice_2020_clean.json",
                    "fashionadvice_2021_clean.json",
                    "fashionadvice_2022_clean.json",
                    "handbags_2019_clean.json",
                    "handbags_2020_clean.json",
                    "handbags_2021_clean.json",
                    "handbags_2022_clean.json",
                    "outfits_2019_clean.json",
                    "outfits_2020_clean.json",
                    "outfits_2021_clean.json",
                    "outfits_2022_clean.json",
                    "petitefashionadvice_2020_clean.json",
                    "petitefashionadvice_2021_clean.json",
                    "petitefashionadvice_2022_clean.json",
                    "plussizefashion_2020_clean.json",
                    "plussizefashion_2021_clean.json",
                    "plussizefashion_2022_clean.json",
                    "weddingdress_2019_clean.json",
                    "weddingdress_2020_clean.json",
                    "weddingdress_2021_clean.json",
                    "weddingdress_2022_clean.json"
    ]
    all_threads = []

    for thread_file in thread_files:
        fp = "../thread_data/threads_to_annotate/comments_5-10/json/" + thread_file
        with open(fp, 'r') as f:
            data = json.load(f)

        all_threads += data['threads']
    
    print(f"Finished merging. Total {len(all_threads)} threads found. Saving updated file to {args.savepath}...")
    with open(args.savepath, 'w') as f:
        json.dump({"threads": all_threads},f,indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--savepath", required=True)
    args = parser.parse_args()
    main(args)