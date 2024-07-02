from argparse import ArgumentParser
import os
import json
from tqdm import tqdm


def main(args):
    corpus_files = os.listdir(args.tilde_corpus_dir)
    corpus_dic = {}
    for file in corpus_files:
        with open(args.tilde_corpus_dir + f"/{file}", 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Loading collection"):
                data = json.loads(line)
                corpus_dic[data['pid']] = data['psg']

    train_files = os.listdir(args.psg_train_dir)
    for file in tqdm(train_files, desc="writing files"):
        id_fout = open(f'{args.output_dir}/{file}', 'a+')
        with open(args.psg_train_dir + f"/{file}", 'r') as f:
            for line in f:
                data = json.loads(line)
                pos_passages = []
                for pos_pass in data["pos"]:
                    pos_passages.append({'pid': pos_pass['pid'],
                                         'passage': corpus_dic[pos_pass['pid']]})
                neg_passages = []
                for neg_pass in data["neg"]:
                    neg_passages.append({'pid': neg_pass['pid'],
                                         'passage': corpus_dic[neg_pass['pid']]})

                temp = {
                    "qry": data['qry'],
                    "pos": pos_passages,
                    "neg": neg_passages
                }
                id_fout.write(f'{json.dumps(temp)}\n')

        id_fout.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--psg_train_dir', required=True)
    parser.add_argument('--tilde_corpus_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)