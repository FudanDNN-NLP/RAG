from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, HfArgumentParser
from modelingv2 import TILDEv2, TILDEv2Trainer
import datasets
import re
import random
from dataclasses import dataclass, field
import os
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from typing import Dict
import torch

@dataclass
class TILDEv2TrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0)
    model_name: str = field(default='bert-base-uncased')
    q_max_len: int = field(default=16)
    p_max_len: int = field(default=192)
    train_group_size: int = field(default=8)
    train_dir: str = field(default=None)
    cache_dir: str = field(default='./cache')
    report_to = []

    def __post_init__(self):
        files = os.listdir(self.train_dir)
        self.train_path = [
            os.path.join(self.train_dir, f)
            for f in files
            if f.endswith('tsv') or f.endswith('json')
        ]

@dataclass
class QryDocCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 16
    max_d_len: int = 192

    def __call__(self, features) -> Dict[str, Dict[str, torch.Tensor]]:
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_d_len,
            return_tensors="pt",
        )

        return {'qry_in': q_collated, 'doc_in': d_collated}


class GroupedMarcoTrainDataset(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(
            self,
            path_to_tsv,
            tokenizer,
            q_max_len,
            p_max_len,
            train_group_size,
            cache_dir,
    ):
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.stop_ids = self.get_stop_ids()
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.train_group_size = train_group_size

        self.nlp_dataset = datasets.load_dataset(
            'json',
            cache_dir=cache_dir,
            data_files=path_to_tsv,
            ignore_verifications=False,
            features=datasets.Features({
                'qry': {
                    'qid': datasets.Value('string'),
                    'query': [datasets.Value('int32')],
                },
                'pos': [{
                    'pid': datasets.Value('string'),
                    'passage': [datasets.Value('int32')],
                }],
                'neg': [{
                    'pid': datasets.Value('string'),
                    'passage': [datasets.Value('int32')],
                }]}
            )
        )['train']
        self.total_len = len(self.nlp_dataset)

    def create_one_example(self, text_encoding, is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            return_attention_mask=False,
            max_length=self.q_max_len if is_query else self.p_max_len,
        )
        return item

    def __len__(self):
        return self.total_len

    def get_stop_ids(self):
        stop_words = set(stopwords.words('english'))
        # keep some common words in ms marco questions
        stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])

        vocab = self.tok.get_vocab()
        tokens = vocab.keys()

        stop_ids = []

        for stop_word in stop_words:
            ids = self.tok(stop_word, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                # bad_token.append(stop_word)
                stop_ids.append(ids[0])

        for token in tokens:
            token_id = vocab[token]
            if token_id in stop_ids:
                continue
            if token == '##s':
                stop_ids.append(token_id)
            if token[0] == '#' and len(token) > 1:
                continue
            if not re.match("^[A-Za-z0-9_-]*$", token):
                # bad_token.append(token)
                stop_ids.append(token_id)

        return set(stop_ids)

    def __getitem__(self, item):
        group = self.nlp_dataset[item]
        group_batch = []
        qid, qry = (group['qry'][k] for k in self.query_columns)
        qry = [id for id in qry if id not in self.stop_ids]

        encoded_query = self.create_one_example(qry, is_query=True)
        _, pos_psg = [
            random.choice(group['pos'])[k] for k in self.document_columns]
        group_batch.append(self.create_one_example(pos_psg))
        if len(group['neg']) < self.train_group_size - 1:
            negs = random.choices(group['neg'], k=self.train_group_size - 1)
        else:
            negs = random.sample(group['neg'], k=self.train_group_size - 1)
        for neg_entry in negs:
            _, neg_psg = [neg_entry[k] for k in self.document_columns]
            group_batch.append(self.create_one_example(neg_psg))

        return encoded_query, group_batch


def main():
    parser = HfArgumentParser(TILDEv2TrainingArguments)
    args: TILDEv2TrainingArguments = parser.parse_args_into_dataclasses()[0]
    config = AutoConfig.from_pretrained(args.model_name,
                                        num_labels=1,
                                        cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              cache_dir=args.cache_dir,
                                              use_fast=False)
    model = TILDEv2.from_pretrained(args.model_name, config=config,
                                    train_group_size=args.train_group_size,
                                    cache_dir=args.cache_dir)
    train_dataset = GroupedMarcoTrainDataset(path_to_tsv=args.train_path,
                                             p_max_len=args.p_max_len,
                                             q_max_len=args.q_max_len,
                                             tokenizer=tokenizer,
                                             train_group_size=args.train_group_size,
                                             cache_dir=args.cache_dir)

    trainer = TILDEv2Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=QryDocCollator(tokenizer, max_q_len=args.q_max_len, max_d_len=args.p_max_len),
    )

    # Training
    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

