import torch
from transformers import BertLMHeadModel, BertTokenizerFast
from tqdm import tqdm
from tools import get_stop_ids, load_run, load_queries
import numpy as np
from timeit import default_timer as timer
import h5py
import argparse
import os
import pickle

# Only has cpu? totally fine with TILDE
DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    queries = load_queries(args.query_path)
    run = load_run(args.run_path)
    model = BertLMHeadModel.from_pretrained(args.ckpt_path, cache_dir=".cache").eval().to(DEVICE)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=".cache")
    stop_ids = get_stop_ids(tokenizer)  # clean the BERT tokenizer vocabulary

    if args.index_path.split('.')[-1] == "pkl":
        with open(args.index_path, 'rb') as handle:
            doc_embeddings = pickle.load(handle)
    elif args.index_path.split('.')[-1] == "hdf5":
        doc_embeddings = {}
        f = h5py.File(args.index_path, 'r')
        doc_file = f['documents']
        for i in tqdm(range(len(doc_file)), desc="loading index....."):
            logs, tids = doc_file[i]
            doc_embeddings[str(i)] = (logs, tids)

    total_tokenizer_time = 0
    total_ranking_time = 0
    lines = []
    for qid in tqdm(run.keys(), desc="Ranking queries...."):
        query = queries[qid]
        docids = run[qid]

        tokenizer_start = timer()
        query_token_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
        cleaned_query_token_ids = [id for id in query_token_ids if id not in stop_ids]  # only keep valid token ids

        if args.alpha != 1:
            query_inputs = tokenizer([query], return_tensors="pt", padding=True, truncation=True)
            query_input_ids = query_inputs["input_ids"]

            query_input_ids[:, 0] = 2  # 2 is the token id for [QRY]

            query_input_ids = query_input_ids.to(DEVICE)
            query_token_type_ids = query_inputs["token_type_ids"].to(DEVICE)
            query_attention_mask = query_inputs["attention_mask"].to(DEVICE)

            with torch.no_grad():
                query_outputs = model(input_ids=query_input_ids,
                                      token_type_ids=query_token_type_ids,
                                      attention_mask=query_attention_mask,
                                      return_dict=True).logits[:, 0]

            query_probs = torch.sigmoid(query_outputs)
            query_log_probs = torch.log10(query_probs)[0].cpu().numpy()

        tokenizer_end = timer()
        total_tokenizer_time += (tokenizer_end - tokenizer_start)

        QL_scores = []
        DL_scores = []

        ranking_start = timer()
        for rank, docid in enumerate(docids):
            if rank == args.cut_off:
                break
            passage_log_probs, passage_token_id = doc_embeddings[docid]
            target_log_probs = passage_log_probs[cleaned_query_token_ids]
            score = np.sum(target_log_probs)
            QL_scores.append(score)

            if args.alpha != 1:
                query_target_log_probs = query_log_probs[passage_token_id]
                passage_score = np.sum(query_target_log_probs) / len(passage_token_id)  # mean likelihood
                DL_scores.append(passage_score)

        if args.alpha != 1:
            scores = (args.alpha * np.array(QL_scores)) + ((1-args.alpha) * np.array(DL_scores))
        else:
            scores = QL_scores

        zipped_lists = zip(scores, docids)
        sorted_pairs = sorted(zipped_lists, reverse=True)

        ranking_end = timer()
        total_ranking_time += (ranking_end - ranking_start)

        num_docs = len(sorted_pairs)

        for i in range(num_docs):
            score, docid = sorted_pairs[i]
            lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(i + 1) + " " + str(score) + " " + f"alpha{args.alpha}" + "\n")
            last_score = score
            last_rank = i
        # add the rest of ranks below cut_off, we don't need to re-rank them.
        for docid in docids[num_docs:]:
            last_score -= 1
            last_rank += 1
            lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(last_rank + 1) + " " + str(
                    last_score) + " " + f"alpha{args.alpha}" + "\n")

    print("Query processing time: %.1f ms" % (1000 * total_tokenizer_time / len(run.keys())))
    print("passage re-ranking time: %.1f ms" % (1000 * total_ranking_time / len(run.keys())))

    with open(args.save_path, "w") as f:
        f.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--cut_off", type=int, default=1000)
    parser.add_argument("--ckpt_path", type=str, default="ielab/TILDE")
    parser.add_argument("--collection_path", type=str, default="./data/collection.tsv")
    args = parser.parse_args()

    if os.path.isdir(args.save_path):
        raise ValueError("save_path requires full path to the output file name")

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main(args)
