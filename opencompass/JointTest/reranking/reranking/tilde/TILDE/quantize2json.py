import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from argparse import ArgumentParser
from timeit import default_timer as timer
import os
import h5py
import json
import sys
from tools import load_queries, load_run, get_stop_ids

class uniform_quantizer:
    def __init__(self, bits, max_impact):
        self.bits = bits
        self.qmax = int(2**bits) - 1
        self.max_impact = max_impact
        print ("Quantizer: [0, ", self.max_impact, "] -> [0, ", self.qmax, "]")

    def quantize(self, score):
        return int((score / self.max_impact) * self.qmax)

def generate_json(docid, vector):
    return json.dumps({"id": docid, "contents": "", "vector": vector}, ensure_ascii=False)
 

def main(args):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, cache_dir="./cache")
    stop_ids = get_stop_ids(tokenizer)
    docids = np.load(os.path.join(args.input_path, "docids.npy"))
    data = h5py.File(os.path.join(args.input_path, "tildev2_index.hdf5"), "r")
    doc_file = data['documents'][:]  # load the hdf5 file to the memory.
    out_file = open(args.output_file, "w", encoding='utf-8')

    assert len(docids) == len(doc_file)
    direct_index = {}

    max_token_impact = 0
    for i, docid in tqdm(enumerate(docids), desc="Creating direct index....."):
        token_scores, token_ids = doc_file[i]
        assert len(token_scores) == len(token_ids)
        direct_index[docid] = {}
        for idx, token_id in enumerate(token_ids):
            tok = tokenizer.convert_ids_to_tokens(int(token_id))
            if tok not in direct_index[docid].keys():
                direct_index[docid][tok] = token_scores[idx]
            else:
                if token_scores[idx] > direct_index[docid][tok]:
                    direct_index[docid][tok] = token_scores[idx]
                    max_token_impact = max(max_token_impact, token_scores[idx])
    del doc_file

    quantizer = uniform_quantizer(args.quantize_bits, max_token_impact)
    for i, docid in tqdm(enumerate(docids), desc="Quantizing and writing json file...."):
        for term in direct_index[docid]:
            score = direct_index[docid][term]
            direct_index[docid][term] = quantizer.quantize(score)
        out_file.write(generate_json(docid, direct_index[docid]))
        out_file.write('\n')
    
    out_file.close()

 
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--quantize-bits", type=int, required=True)
    args = parser.parse_args()

    if args.quantize_bits < 1 or args.quantize_bits > 16:
        print("--quantize-bits should be in the range [1,16]")
        sys.exit(-1)

    main(args)
