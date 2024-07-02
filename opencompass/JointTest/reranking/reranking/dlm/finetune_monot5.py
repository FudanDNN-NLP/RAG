import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0*")


class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        # return text, sample[2]
        # only 'input_ids' is recognized by Trainer()
        return {
          'input_ids': text,
          'labels': sample[2],
        }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='t5-base', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--triples_path", default='../data/msmarco/passage/cn/triples.train.ids.small.tsv', type=str, required=False,
                        help="triples.tsv path")
    parser.add_argument("--collection_path", default='../data/msmarco/passage/cn/collection.tsv', type=str, required=False,
                        help="collection.tsv path, if triples is ids")
    parser.add_argument("--queries_path", default='../data/msmarco/passage/cn/queries.train.tsv', type=str, required=False,
                        help="queries.tsv path, if triples is ids")
    parser.add_argument("--output_model_path", default='models/T5-base_zh/', type=str, required=False,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=1000, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=100, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=16, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=1, type=int, required=False,
                        help="Number of epochs to train")

    print("Loading Model & Tokenizer ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    # model = torch.nn.DataParallel(model, device_ids=[6, 7])
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("Loading Training Samples ...")
    train_samples = []
    if 'ids' not in args.triples_path:
        with open(args.triples_path, 'r', encoding="utf-8") as fIn:
            for num, line in enumerate(fIn):
                query, positive, negative = line.split("\t")
                train_samples.append((query, positive, 'true'))
                train_samples.append((query, negative, 'false'))
    else:
        docs, queries = {}, {}
        with open(args.collection_path, 'r', encoding="utf-8") as fIn:  # 8841823
            for num, line in tqdm(enumerate(fIn), desc="Loading collection..."):
                did, content = line.strip().split("\t")
                docs[did] = content
        with open(args.queries_path, 'r', encoding="utf-8") as fIn:     # 808731
            for num, line in tqdm(enumerate(fIn), desc="Loading queries..."):
                qid, content = line.strip().split("\t")
                queries[qid] = content
        with open(args.triples_path, 'r', encoding="utf-8") as fIn:     # 39780811
            for num, line in tqdm(enumerate(fIn), desc="Loading triples..."):
                # if num == 10000000:
                #     break
                qid, pos_did, neg_did = line.strip().split("\t")
                train_samples.append((queries[qid], docs[pos_did], 'true'))
                train_samples.append((queries[qid], docs[neg_did], 'false'))
        del docs, queries


    def smart_batching_collate_text_only(batch):
        texts = [example['input_ids'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']
        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)
        return tokenized

    dataset_train = MonoT5Dataset(train_samples)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=5e-5,
        do_train=True,
        save_strategy=strategy,
        save_steps=steps,
        logging_steps=args.logging_steps,
        warmup_steps=500,
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
    )

    # trainer.train()
    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(args.output_model_path)
    trainer.save_state()


if __name__ == "__main__":
    main()