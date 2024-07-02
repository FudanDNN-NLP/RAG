import torch
from transformers import BertLMHeadModel, BertTokenizerFast
from tqdm import tqdm
from tools import get_stop_ids, load_run, load_collection, get_batch_text
import numpy as np
import pickle
import h5py
import argparse
import os
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args):
    collection = load_collection(args.collection_path)
    if args.run_path:
        run = load_run(args.run_path)
        doc_embeddings = {}
        docids = []
        for qid in run.keys():
            docids.extend(run[qid])
        docids = list(set(docids))
    else:
        docids = list(collection.keys())
        dt_token_id = h5py.vlen_dtype(np.dtype('int16'))
        dt_embedding = np.dtype((np.float16, (30522,)))
        dt_compound = np.dtype([('embedding', dt_embedding), ('token_ids', dt_token_id)])
        f = h5py.File(os.path.join(args.output_path, "passage_embeddings.hdf5"), "w")

    model = BertLMHeadModel.from_pretrained(args.ckpt_path_or_name, cache_dir=".cache").eval().to(DEVICE)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=".cache")
    stop_ids = get_stop_ids(tokenizer)


    batch_size = args.batch_size
    num_docs = len(docids)
    num_iter = num_docs // batch_size + 1

    if not args.run_path:
        dset = f.create_dataset("documents", (num_docs,), dtype=dt_compound)

    docno = 0
    for i in tqdm(range(num_iter), desc="Indexing passages"):
        start = i * batch_size
        end = (i + 1) * batch_size
        if end > num_docs:
            end = num_docs
            if start == end:
                continue

        batch_text = get_batch_text(start, end, docids, collection)
        passage_token_ids = tokenizer(batch_text, add_special_tokens=False)["input_ids"]
        cleaned_ids = []
        for passage_token_id in passage_token_ids:
            cleaned_passage_token_id = [id for id in passage_token_id if id not in stop_ids]
            cleaned_passage_token_id = np.array(cleaned_passage_token_id).astype(np.int16)
            cleaned_ids.append(cleaned_passage_token_id)
        cleaned_passage_token_ids = cleaned_ids
        passage_inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True)
        passage_input_ids = passage_inputs["input_ids"]
        passage_input_ids[:, 0] = 1  # 1 is the token id for [DOC]
        passage_token_type_ids = passage_inputs["token_type_ids"].to(DEVICE)
        passage_input_ids = passage_input_ids.to(DEVICE)
        passage_attention_mask = passage_inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            passage_outputs = model(input_ids=passage_input_ids,
                                    token_type_ids=passage_token_type_ids,
                                    attention_mask=passage_attention_mask,
                                    return_dict=True).logits[:, 0]
            passage_probs = torch.sigmoid(passage_outputs)
            passage_log_probs = torch.squeeze(torch.log10(passage_probs)).cpu().numpy().astype(np.float16)

        for inbatch_id, docid in enumerate(docids[start:end]):
            if args.run_path:
                doc_embeddings[docid] = (passage_log_probs[inbatch_id], cleaned_passage_token_ids[inbatch_id])
            else:
                assert str(docno) == docid
                doc = (passage_log_probs[inbatch_id], cleaned_passage_token_ids[inbatch_id])
                dset[docno] = doc
                docno += 1

    if args.run_path:
        with open(os.path.join(args.output_path, "passage_embeddings.pkl"), 'wb') as handle:
            pickle.dump(doc_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path_or_name", type=str, default="ielab/TILDE")
    parser.add_argument("--run_path", type=str)
    parser.add_argument("--collection_path", type=str, default="./data/collection/collection.tsv")
    parser.add_argument("--output_path", type=str, default="./data/index/TILDE")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    main(args)
