import torch
import argparse
from tqdm import tqdm
import numpy as np
import json
import h5py
import os
from .tools import get_stop_ids
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
from .modelingv2 import TILDEv2
from torch.utils.data import Dataset, DataLoader


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MsmarcoDataset(Dataset):
    def __init__(self, collection_path: str, tokenizer: PreTrainedTokenizer, p_max_len=192):
        self.collection = []
        self.docids = []

        # Use Expanded collection
        if os.path.isdir(collection_path):
            for filename in os.listdir(collection_path):
                with open(f"{collection_path}/{filename}", 'r') as f:
                    lines = f.readlines()
                    for line in tqdm(lines, desc="loading collection...."):
                        data = json.loads(line)
                        self.collection.append(data['psg'])
                        self.docids.append(data['pid'])
        # Use original msmarco collection
        else:
            with open(f"{collection_path}", 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc="loading collection...."):
                    pid, psg = line.strip().split("\t")
                    self.collection.append(psg)
                    self.docids.append(pid)


        self.tok = tokenizer
        self.p_max_len = p_max_len

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, item) -> [BatchEncoding, BatchEncoding]:
        psg = self.collection[item]
        encoded_psg = self.tok.encode_plus(
            psg,
            max_length=self.p_max_len,
            truncation='only_first',
            return_attention_mask=False,
        )
        return encoded_psg

    def get_docids(self):
        return self.docids


def main(model, tokenizer, args):

    #tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path_or_name, use_fast=False, cache_dir='./cache')
    #model = TILDEv2.from_pretrained(args.ckpt_path_or_name, cache_dir='./cache').eval().to(DEVICE)
    # model = TILDEv2.from_pretrained(args.ckpt_path_or_name, cache_dir='./cache').eval()
    # model = torch.nn.DataParallel(model)
    # model = model.cuda()

    sepcial_token_ids = tokenizer.all_special_ids
    stop_ids = get_stop_ids(tokenizer)
    stop_ids = stop_ids.union(sepcial_token_ids)  # add bert special token ids as well.

    dataset = MsmarcoDataset(
        args['collection_path'], tokenizer, p_max_len=args['p_max_len'],
    )
    docids = dataset.get_docids()
    np.save(os.path.join(args['output_path'], "docids.npy"), np.array(docids))

    data_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        collate_fn=DataCollatorWithPadding(
            tokenizer,
            max_length=args['p_max_len'],
            padding='max_length'
        ),
        shuffle=False,    # important
        drop_last=False,  # important
        num_workers=args['num_workers'],
    )

    dt_token_id = h5py.vlen_dtype(np.dtype('int16'))
    dt_embedding = h5py.vlen_dtype(np.dtype('float16'))
    dt_compound = np.dtype([('embedding', dt_embedding), ('token_ids', dt_token_id)])
    f = h5py.File(os.path.join(args['output_path'], "tildev2_index.hdf5"), "w")
    dset = f.create_dataset("documents", (len(docids),), dtype=dt_compound)

    docno = 0
    for passage_inputs in tqdm(data_loader):
        passage_token_ids = passage_inputs["input_ids"].cpu().numpy().astype(np.int16)
        passage_inputs.to(model.device)
        # passage_inputs.cuda()

        with torch.no_grad():
            passage_outputs = model.encode(**passage_inputs)
            passage_outputs = passage_outputs.squeeze(1).cpu().numpy().astype(np.float16)

        for inbatch_idx in range(len(passage_token_ids)):
            token_scores = []
            token_ids = []
            for idx, token_id in enumerate(passage_token_ids[inbatch_idx]):
                if token_id in stop_ids:
                    continue
                score = passage_outputs[inbatch_idx][idx]
                token_scores.append(score)
                token_ids.append(token_id)
            token_scores = np.array(token_scores, dtype=np.float16)
            token_ids = np.array(token_ids, dtype=np.int16)
            doc = (token_scores, token_ids)
            dset[docno] = doc
            docno += 1

    f.close()

