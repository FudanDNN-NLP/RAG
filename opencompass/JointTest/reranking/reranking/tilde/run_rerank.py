import os
import time
from argparse import ArgumentParser
from timeit import default_timer as timer
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import h5py
import json

from .TILDE.expansion import main as expansion_main
from .TILDE.indexingv2 import main as indexing_main
from .TILDE.tools import get_stop_ids


cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
if not os.path.exists(cur_dir + "runs/temp"):
    os.makedirs(cur_dir + "runs/temp")
corpus_path = cur_dir + "runs/temp/collection.jsonl"
expand_path = cur_dir + "runs/temp/expanded"
index_path = cur_dir + "runs/temp/index/TILDEv2"
result_path = cur_dir + "runs/temp/TILDEv2.txt"


def tilde_rerank(model, expansion_model, tokenizer, bert_tokenizer, query, docs):

    create_temp_collection(docs)
    start_time = time.time()
    passage_expansion(expansion_model, bert_tokenizer)
    index_collection(model, tokenizer)
    end_time = time.time()
    with open(cur_dir + "runs/temp/time.out", "a") as f:
        f.write(str(end_time - start_time) + "\n")
    reranked_docs, scores = execute_rerank(bert_tokenizer, query, docs)
    print(f"Index time: {end_time - start_time}s")

    return reranked_docs, scores


def create_temp_collection(docs):
    with open(corpus_path, "w") as file:
        for index, content in enumerate(docs):
            # file.write(f"{index}\t{content}\n")
            file.write(json.dumps({"index": index, "content": content}) + "\n")
    print("Converted docs to temp collection")


def passage_expansion(expansion_model, bert_tokenizer):

    args = {"corpus_path": corpus_path, "output_dir": expand_path, "topk": 200, "batch_size": 64, "num_workers": 8, "store_raw": True}

    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    expansion_main(expansion_model, bert_tokenizer, args)
    print("Completed passage expansion")


def index_collection(model, tokenizer):

    args = {
        "ckpt_path_or_name": "ielab/TILDEv2-TILDE200-exp",
        "collection_path": expand_path,
        "output_path": index_path,
        "batch_size": 256,
        "num_workers": 4,
        "p_max_len": 192,
    }

    if not os.path.exists(args["output_path"]):
        os.makedirs(args["output_path"])

    indexing_main(model, tokenizer, args)
    print("Completed indexing collection")


def execute_rerank(tokenizer, query, docs):
    run_type = "msmarco"

    stop_ids = get_stop_ids(tokenizer)

    print("Loading hdf5 file.....")

    f = h5py.File(os.path.join(index_path, "tildev2_index.hdf5"), "r")
    doc_file = f["documents"][:]  # load the hdf5 file to the memory.
    doc_ids = np.load(os.path.join(index_path, "docids.npy"))

    assert len(doc_ids) == len(doc_file)
    direct_index = {}

    for i, doc_id in tqdm(enumerate(doc_ids), desc="Creating direct index....."):
        doc_id = str(doc_id)
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

    print("Reranking.....")

    # Tokenizing
    tokenizer_start = timer()
    query_token_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
    cleaned_query_token_ids = [id for id in query_token_ids if id not in stop_ids]  # remove stopwords for query
    tokenizer_end = timer()
    total_tokenizer_time += tokenizer_end - tokenizer_start

    # Re-ranking
    ranking_start = timer()
    scores = []
    # print(direct_index)
    doc_ids = [str(i) for i in range(len(docs))]  # ['0', '1', ... 'n']
    for rank, doc_id in enumerate(doc_ids):
        token_scores = direct_index[doc_id]
        doc_score = 0
        for token_id in cleaned_query_token_ids:
            if token_id in token_scores.keys():
                doc_score += token_scores[token_id].item()
        scores.append(doc_score)
    zipped_lists = zip(scores, doc_ids[: len(scores)])
    sorted_pairs = sorted(zipped_lists, reverse=True)

    ranking_end = timer()
    total_ranking_time += ranking_end - ranking_start

    num_docs = len(sorted_pairs)
    print("Reranked (score, doc_id): " + str(sorted_pairs))
    reranked_docs, scores = [], []
    for i in range(num_docs):
        score, doc_id = sorted_pairs[i]
        reranked_docs.append(docs[int(doc_id)])
        scores.append(score)
        if run_type == "msmarco":
            lines.append("query" + "\t" + str(doc_id) + "\t" + str(i + 1) + "\n")
        else:
            lines.append("query" + " " + "Q0" + " " + str(doc_id) + " " + str(i + 1) + " " + str(score) + " " + "TILDEv2" + "\n")

    # print("Avg query processing time: %.1f ms" % (1000 * total_tokenizer_time))
    # print(f"Avg passage reranking time: %.1f ms" % (1000 * total_ranking_time))

    with open(result_path, "w") as f:
        f.writelines(lines)

    return reranked_docs, scores
