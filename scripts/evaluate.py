import argparse
from rouge import FilesRouge
import subprocess

def main(args):
    # with open(args.reference, 'r') as f:
        # refs = f.readlines()
    files_rouge = FilesRouge()
    print("Computing ROUGE score...")
    scores = files_rouge.get_scores(ref_path=args.reference, hyp_path=args.hypothesis, avg=True)
    print(f"R1: {scores['rouge-1']['f']} \n R2: {scores['rouge-2']['f']} \n RL: {scores['rouge-l']['f']}")
    print("Computing BertScore...")
    subprocess.run(f"bert-score -r {args.reference} -c {args.hypothesis} --lang en --rescale_with_baseline" ,shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hyp', '--hypothesis', type=str, help="Name of file with hypotheses.")
    parser.add_argument('-ref', '--reference', type=str, help="Name of file with references.")
    # parser.add_argument('-s', '--savefile', type=str, help="Name of file to save to. Should be a jsonfile")
    args = parser.parse_args()
    main(args)