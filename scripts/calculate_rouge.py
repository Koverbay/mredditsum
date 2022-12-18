import os
import json
import argparse
from rouge import Rouge

def main(args):

    with open(args.f, 'r') as f:
        data = json.load(f)

    threads = data['threads']
    refs = list()
    hyps_all = {}
    for hypfield in args.hypfields:
        hyps_all[hypfield] = []
    
    for thread in threads:
        refs.append(thread[args.reffield])
        for hypfield in args.hypfields:
            hyps_all[hypfield].append(thread[hypfield])
    
    rouge = Rouge()
    scores = {}
    for hypfield in args.hypfields:
        hyps = hyps_all[hypfield]
        scores[hypfield] = rouge.get_scores(hyps, refs, avg=True)

    print(scores)

    score_data = {f'rouge_fashionadvice_2022-07': scores}
    with open('rougedata.json', 'w') as f:
        json.dump(score_data, f, indent=4)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True, type=str)
    parser.add_argument('-reffield', type=str, default="final_summary")
    parser.add_argument('-hypfields', nargs='+', required=True)
    args = parser.parse_args()
    main(args)