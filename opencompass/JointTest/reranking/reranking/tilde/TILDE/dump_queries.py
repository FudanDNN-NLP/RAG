import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from argparse import ArgumentParser
from timeit import default_timer as timer
import os
import h5py
from tools import load_queries, load_run, get_stop_ids


def main(args):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, cache_dir="./cache")
    stop_ids = get_stop_ids(tokenizer)

    queries = load_queries(args.query_path)

    for qid in queries.keys():
        query = queries[qid]
        query_token_ids = tokenizer(query, add_special_tokens=False)["input_ids"]
        cleaned_query_token_ids = [id for id in query_token_ids if id not in stop_ids]  # remove stopwords for query
        tok = tokenizer.convert_ids_to_tokens(cleaned_query_token_ids)
        print (qid + "\t" + " ".join(tok))
 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--query_path', type=str, required=True)
    args = parser.parse_args()

    main(args)
