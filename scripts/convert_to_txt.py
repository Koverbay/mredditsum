import json
import argparse


def main(args):

    fp = args.filename
    src_savepath = fp[:len(fp)-5] + '_src.txt'
    tgt_savepath = fp[:len(fp)-5] + '_tgt.txt'
    eval_savepath = fp[:len(fp)-5] + '_eval.txt'

    src_data = []
    tgt_data = []
    eval_data = []

    with open(fp, 'r') as f:
        data = json.load(f)

    for item in data['threads']:
        item_id = item['id']
        item_src = item['document']
        item_tgt = item['summary']
        if args.hiert5:
            src_data.append(item_src)
            tgt_data.append(item_tgt)
            eval_data.append(item_tgt)
        else:
            src_data.append(item_id + ' ' + item_src)
            tgt_data.append(item_id + ' ' + item_tgt)
            eval_data.append(item_tgt)

    print(f"Saving source data to {src_savepath}...")

    with open(src_savepath, 'w') as f:
        for i in src_data:
            if i == src_data[-1]:
                f.write(i)
            else:
                f.write(i+'\n')

    print(f"Saving target data to {tgt_savepath}...")

    with open(tgt_savepath, 'w') as f:
        for i in tgt_data:
            if i == tgt_data[-1]:
                f.write(i)
            else:
                f.write(i+'\n')

    print(f"Saving eval data to {eval_savepath}...")
    with open(eval_savepath, 'w') as f:
        for i in eval_data:
            if i == eval_data[-1]:
                f.write(i)
            else:
                f.write(i+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', type=str, help="Name of processed json file with data.")
    parser.add_argument('-ht', '--hiert5', action='store_true')
    args = parser.parse_args()
    main(args)