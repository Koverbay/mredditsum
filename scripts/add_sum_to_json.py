import json
import argparse

# Adds the annotations from the MTURK json file to our original data json file.

def main(args):
    # Load json files

    with open(args.filepath_opsum, 'r') as f:
        opsum_data = json.load(f)

    with open(args.filepath_csum, 'r') as f:
        csum_data = json.load(f)

    with open(args.filepath_original, 'r') as f:
        original_data = json.load(f)
    
    new_threads = []
    
    # For each post in data file, get the matching one from the MTURK file
    # Add in the opsum or cluster sum
    for thread in original_data['threads']:
        unedited_sum = ""
        submission_id = thread['submission_id']
        if (submission_id in opsum_data) and (submission_id in csum_data):
            opsum = opsum_data[submission_id]['Answer.opsum']
            thread['opsum'] = opsum
            unedited_sum += opsum
            # Append all the cluster summaries
        
            csums = []
            for i in range(6):
                csum = csum_data[submission_id][f'Answer.clustersum{i}']
                # Only append non-empty summaries
                if (csum != "{}") and (csum != ""):
                    csums.append(csum)
                    unedited_sum += f" {csum}"
            thread['csums'] = csums
            thread['unedited_sum'] = unedited_sum
            new_threads.append(thread)
        else:
            print(f"Missing datapoint for opsum: {submission_id}")


        

    new_data = {'threads': new_threads}
    # Save updated file
    savepath = args.filepath_original[:(len(args.filepath_original)-5)] + "_annotated.json"
    print(f"Finished updating annotations. Saving data to {savepath}...")
    with open(savepath, 'w') as f:
        json.dump(new_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fopsum", "--filepath_opsum", required=True)
    parser.add_argument("-fcsum", "--filepath_csum", required=True)
    parser.add_argument("-forig", "--filepath_original", required=True)
    args = parser.parse_args()
    main(args)