import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from argparse import ArgumentParser
from timeit import default_timer as timer
import os
import h5py
from TILDE.tools import load_queries, load_run, get_stop_ids


def main(args):
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, cache_dir="./cache")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    stop_ids = get_stop_ids(tokenizer)

    queries = load_queries(args.query_path)
    run = load_run(args.run_path, run_type=args.run_type)

    print("loading hdf5 file.....")

    f = h5py.File(os.path.join(args.index_path, "tildev2_index.hdf5"), 'r')
    doc_file = f['documents'][:]  # load the hdf5 file to the memory.
    doc_ids = np.load(os.path.join(args.index_path, "docids.npy"))

    assert len(doc_ids) == len(doc_file)
    direct_index = {}

    for i, doc_id in tqdm(enumerate(doc_ids), desc="Creating direct index....."):
        token_scores, token_ids = doc_file[i]
        assert len(token_scores) == len(token_ids)
        direct_index[doc_id] = {}
        for idx, token_id in enumerate(token_ids):
            if token_id not in direct_index[doc_id].keys():
                direct_index[doc_id][token_id] = token_scores[idx]
            else:
                if token_scores[idx] > direct_index[doc_id][token_id]:
                    direct_index[doc_id][token_id] = token_scores[idx]
    del doc_file

    total_tokenizer_time, total_ranking_time = 0, 0
    lines = []

    if len(run.keys()) < len(queries.keys()):
        qids = run.keys()
    else:
        qids = queries.keys()

    for qid in tqdm(qids, desc="Re-ranking queries...."):
        query = queries[qid]
        doc_ids = run[qid]

        # Tokenizing time.
        tokenizer_start = timer()

        query_token_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
        cleaned_query_token_ids = [id for id in query_token_ids if id not in stop_ids]  # remove stopwords for query

        tokenizer_end = timer()
        total_tokenizer_time += tokenizer_end - tokenizer_start

        # Re-ranking time.
        ranking_start = timer()

        scores = []
        for rank, doc_id in enumerate(doc_ids):
            if rank == args.cut_off:
                break
            token_scores = direct_index[doc_id]
            doc_score = 0
            for token_id in cleaned_query_token_ids:
                if token_id in token_scores.keys():
                    doc_score += (token_scores[token_id].item())
            scores.append(doc_score)
        zipped_lists = zip(scores, doc_ids[:len(scores)])
        sorted_pairs = sorted(zipped_lists, reverse=True)

        ranking_end = timer()
        total_ranking_time += ranking_end - ranking_start

        # We dont count writing to the total time
        num_docs = len(sorted_pairs)
        last_score, last_rank = 0, 0
        for i in range(num_docs):
            score, doc_id = sorted_pairs[i]
            if args.run_type == 'msmarco':
                lines.append(str(qid) + "\t" + str(doc_id) + "\t" + str(i + 1) + "\n")
            else:
                lines.append(str(qid) + " " + "Q0" + " " + str(doc_id) + " " + str(i + 1) + " " + str(score) + " " + "TILDEv2" + "\n")
            last_score = score
            last_rank = i

        # add the rest of ranks below cut_off, we don't need to re-rank them.
        for doc_id in doc_ids[num_docs:]:
            last_score -= 1
            last_rank += 1
            if args.run_type == 'msmarco':
                lines.append(str(qid) + "\t" + str(doc_id) + "\t" + str(last_rank + 1) + "\n")
            else:
                lines.append(str(qid) + " " + "Q0" + " " + str(doc_id) + " " + str(last_rank + 1) + " " + str(last_score) + " " + "TILDEv2" + "\n")

    print("Avg query processing time: %.1f ms" % (1000 * total_tokenizer_time / len(run.keys())))
    print(f"Avg passage re-ranking top{args.cut_off} time: %.1f ms" % (1000 * total_ranking_time / len(run.keys())))

    with open(args.save_path, "w") as f:
        f.writelines(lines)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--index_path', type=str, default='TILDE/data/index/TILDEv2')
    parser.add_argument('--query_path', type=str, default='../data/msmarco_ans_small/queries.dev.small.tsv')
    parser.add_argument('--run_path', type=str, default='../data/msmarco_ans_small/run.dev.small.tsv')
    parser.add_argument('--save_path', type=str, default='runs/TILDEv2.txt')
    parser.add_argument('--run_type', type=str, help="msmarco or trec", default='msmarco')
    parser.add_argument('--cut_off', type=int, default=1000)
    args = parser.parse_args()

    if os.path.isdir(args.save_path):
        raise ValueError("save_path requires full path to the output file name")

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main(args)
