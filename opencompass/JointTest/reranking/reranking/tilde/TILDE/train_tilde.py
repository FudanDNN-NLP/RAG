import pytorch_lightning as pl
from transformers import BertTokenizer
import torch
from tools import get_stop_ids
import random
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from modeling import TILDE
import os


MODEL_TYPE = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir='./cache')


class CheckpointEveryEpoch(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        start_epoc,
        save_path,
    ):

        self.start_epoc = start_epoc
        self.file_path = save_path

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = trainer.current_epoch
        if epoch >= self.start_epoc:
            ckpt_path = os.path.join(self.save_path, f"epoch_{epoch+1}.ckpt")
            trainer.save_checkpoint(ckpt_path)


class MsmarcoDocumentQueryPair(Dataset):
    def __init__(self, path):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE, cache_dir='./cache')
        self.path = path
        self.queries = []
        self.passages = []
        self.stop_ids = list(get_stop_ids(self.tokenizer))

        with open(path, 'r') as f:
            contents = f.readlines()

        for line in contents:
            passage, query = line.strip().split('\t')

            self.queries.append(query)
            self.passages.append(passage)

    def __getitem__(self, index):
        query = self.queries[index]
        passage = self.passages[index]

        ind = self.tokenizer(query, add_special_tokens=False)['input_ids']
        cleaned_ids = []
        for id in ind:
            if id not in self.stop_ids:
                cleaned_ids.append(id)
        yq = torch.zeros(self.tokenizer.vocab_size, dtype=torch.float32)
        yq[cleaned_ids] = 1
        yq[self.stop_ids] = -1

        ind = self.tokenizer(passage, add_special_tokens=False)['input_ids']
        cleaned_ids = []
        for id in ind:
            if id not in self.stop_ids:
                cleaned_ids.append(id)
        yd = torch.zeros(self.tokenizer.vocab_size, dtype=torch.float32)
        yd[cleaned_ids] = 1
        yd[self.stop_ids] = -1

        return passage, yq, query, yd

    def __len__(self):
        return len(self.queries)


def make_negative_labels(ys):
    batch_size = len(ys)
    neg_ys = []
    for i in range(batch_size):
        weigths = [1/(batch_size-1)] * batch_size
        weigths[i] = 0
        neg_ys.append(random.choices(ys, weights=weigths)[0])
    return neg_ys


def collate_fn(batch):
    passages = []
    queries = []
    yqs = []
    yds = []

    for passage, yq, query, yd in batch:
        passages.append(passage)
        yqs.append(yq)
        queries.append(query)
        yds.append(yd)

    passage_inputs = tokenizer(passages, return_tensors="pt", padding=True, truncation=True)
    passage_input_ids = passage_inputs["input_ids"]
    passage_token_type_ids = passage_inputs["token_type_ids"]
    passage_attention_mask = passage_inputs["attention_mask"]
    neg_yqs = make_negative_labels(yqs)

    query_inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
    query_input_ids = query_inputs["input_ids"]
    query_token_type_ids = query_inputs["token_type_ids"]
    query_attention_mask = query_inputs["attention_mask"]
    neg_yds = make_negative_labels(yds)

    passage_input_ids[:, 0] = 1  # 1 is token id for [DOC]
    query_input_ids[:, 0] = 2   # 2 is token id for [QRY]

    return passage_input_ids, passage_token_type_ids, passage_attention_mask, torch.stack(yqs), torch.stack(neg_yqs), \
           query_input_ids, query_token_type_ids, query_attention_mask, torch.stack(yds), torch.stack(neg_yds)


def main(args):
    seed_everything(313)
    tb_logger = pl_loggers.TensorBoardLogger('logs/'.format(MODEL_TYPE))

    model = TILDE(MODEL_TYPE, gradient_checkpointing=args.gradient_checkpoint)
    dataset = MsmarcoDocumentQueryPair(args.train_path)
    loader = DataLoader(dataset,
                        batch_size=128,
                        drop_last=True,
                        pin_memory=True,
                        shuffle=True,
                        num_workers=10,
                        collate_fn=collate_fn)

    trainer = Trainer(max_epochs=10,
                      gpus=1,
                      checkpoint_callback=False,
                      logger=tb_logger,
                      # accelerator="ddp",
                      # plugins='ddp_sharded',
                      callbacks=[CheckpointEveryEpoch(0, args.save_path)]
                      )
    trainer.fit(model, loader)
    print("Saving the final checkpoint as a huggingface model...")
    model_to_save = TILDE.load_from_checkpoint(model_type=MODEL_TYPE, checkpoint_path=os.path.join(args.save_path, 'epoch_10.ckpt'))
    model_to_save.save(os.path.join(args.save_path, 'TILDE'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--gradient_checkpoint", action='store_true', help='Ture for trade off training speed for larger batch size')
    args = parser.parse_args()

    main(args)
