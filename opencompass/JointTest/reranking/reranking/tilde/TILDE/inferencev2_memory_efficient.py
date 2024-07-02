import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from argparse import ArgumentParser
from timeit import default_timer as timer
import os
import h5py
from tools import load_queries, load_run, get_stop_ids


def get_token_weights(qry_token_ids, docid, index):
    token_scores, token_ids = index[docid]
    token_idx = [np.where(token_ids == id) for id in qry_token_ids]
    return_scores = []
    for idx in token_idx:
        if len(idx[0]) != 0:
            return_scores.append(np.max(token_scores[idx]))
    return return_scores


def main(args):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, cache_dir="./cache")
    stop_ids = get_stop_ids(tokenizer)

    queries = load_queries(args.query_path)
    run = load_run(args.run_path, run_type=args.run_type)

    print("loading hdf5 file.....")

    f = h5py.File(os.path.join(args.index_path, "tildev2_index.hdf5"), 'r')
    doc_file = f['documents'][:]  # load the hdf5 file to the memory.
    all_docids = np.load(os.path.join(args.index_path, "docids.npy"))
    docid_map = {}
    for i, docid in tqdm(enumerate(all_docids), desc="Creating internal docid map....."):
        docid_map[docid] = i

    assert len(all_docids) == len(doc_file)
    direct_index = doc_file
    total_tokenizer_time = 0
    total_ranking_time = 0
    lines = []

    if len(run.keys()) < len(queries.keys()):
        qids = run.keys()
    else:
        qids = queries.keys()

    for qid in tqdm(qids, desc="Re-ranking queries...."):
        query = queries[qid]
        docids = run[qid]

        # Tokenizing time.
        tokenizer_start = timer()

        query_token_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
        cleaned_query_token_ids = [id for id in query_token_ids if id not in stop_ids]  # remove stopwords for query

        tokenizer_end = timer()
        total_tokenizer_time += (tokenizer_end - tokenizer_start)

        # Re-ranking time.
        ranking_start = timer()

        scores = []
        for rank, docid in enumerate(docids):
            if rank == args.cut_off:
                break
            doc_score = np.sum(get_token_weights(cleaned_query_token_ids, docid_map[docid], direct_index))
            scores.append(doc_score)

        zipped_lists = zip(scores, docids[:len(scores)])
        sorted_pairs = sorted(zipped_lists, reverse=True)

        ranking_end = timer()
        total_ranking_time += (ranking_end - ranking_start)

        # We dont count writing to the total time
        num_docs = len(sorted_pairs)
        last_score = 0
        last_rank = 0
        for i in range(num_docs):
            score, docid = sorted_pairs[i]
            if args.run_type == 'msmarco':
                lines.append(str(qid) + "\t" + str(docid) + "\t" + str(i + 1) + "\n")
            else:
                lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(i + 1) + " " + str(
                    score) + " " + "TILDEv2" + "\n")
            last_score = score
            last_rank = i

        # add the rest of ranks below cut_off, we don't need to re-rank them.
        for docid in docids[num_docs:]:
            last_score -= 1
            last_rank += 1
            if args.run_type == 'msmarco':
                lines.append(str(qid) + "\t" + str(docid) + "\t" + str(last_rank + 1) + "\n")
            else:
                lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(last_rank + 1) + " " + str(
                    last_score) + " " + "TILDEv2" + "\n")

    print("Avg query processing time: %.1f ms" % (1000 * total_tokenizer_time / len(run.keys())))
    print(f"Avg passage re-ranking top{args.cut_off} time: %.1f ms" % (1000 * total_ranking_time / len(run.keys())))

    with open(args.save_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--query_path', type=str, required=True)
    parser.add_argument('--run_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--run_type', type=str, help="msmarco or trec", default='trec')
    parser.add_argument('--cut_off', type=int, default=1000)
    args = parser.parse_args()

    if os.path.isdir(args.save_path):
        raise ValueError("save_path requires full path to the output file name")

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main(args)
