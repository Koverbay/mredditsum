import json
import argparse
import glob
import pdb

# Adds the cluster information to the data

def main(args):
    # Get all of our thread data
    with open(args.filepath, 'r') as f:
        all_threads = json.load(f)['threads']

    # Get all the AMT data together, key'd  by id
    amt_files = glob.glob("/gallery_tate/keighley.overbay/thread_summarization/annotation_batches/*/*_csum_results.json")
    all_amt_data = {}
    for amt_file in amt_files:
        with open(amt_file,'r') as f:
            amt_data = json.load(f)
        all_amt_data = all_amt_data | amt_data

    # Then for each of our points, get the AMT info and mark the appropriate clusters
    
    new_threads = []
    for thread in all_threads:
        submission_id = thread['submission_id']

        if (submission_id in all_amt_data):
            
            sumorskips = []
            summed = []
            raw_csums = []
            for i in range(6):
                sumorskip = all_amt_data[submission_id][f'Answer.sumorskip{i}']
                sumorskips.append(sumorskip)
                if sumorskip.startswith('sum'):
                    raw_csum = all_amt_data[submission_id][f'Answer.clustersum{i}']
                    raw_csums.append(raw_csum)
            i = 0
            for cluster_id in thread['clusters_auto']:
                if sumorskips[i].startswith('sum'):
                    summed.append(cluster_id)
                # sumorskipped[cluster_id] = sumorskips[i]
                i+=1
                if i > 5:
                    break
            
            csums_with_ids = {}
            if len(thread['csums']) != len(summed):
                print(f"Issue with summed, csum length mismatch on point {submission_id}")
                print(f"Using raw summaries...")
                for i in range(len(summed)):
                    cluster_id=summed[i]
                    csums_with_ids[cluster_id] =raw_csums[i]
                thread['csums_with_ids'] = csums_with_ids
                new_threads.append(thread)
                # with open('ids_to_check.txt', 'a') as f:
                #     f.write(submission_id)
                #     f.write('\n')
                # pdb.set_trace()
                continue

            for i, csum in enumerate(thread['csums']):
                cluster_id = summed[i]
                csums_with_ids[cluster_id] = csum
                
            thread['csums_with_ids'] = csums_with_ids
            new_threads.append(thread)
        else:
            print(f"Missing datapoint for id: {submission_id}")


    new_data = {'threads': new_threads}
    # Save updated file
    savepath = args.savepath
    print(f"Finished updating annotations. Saving data to {savepath}...")
    with open(savepath, 'w') as f:
        json.dump(new_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--savepath", required=True)
    parser.add_argument("-f", "--filepath", required=True)
    # parser.add_argument("-fcsum", "--filepath_csum", required=True)
    # parser.add_argument("-forig", "--filepath_original", required=True)
    args = parser.parse_args()
    main(args)