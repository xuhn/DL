import json
import os
from typing import Optional, Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils import data
import torch
from dataset import Vocab, BertDataSet
from transformers import AutoTokenizer

class NERDataModule(pl.LightningDataModule):

    def __init__(self, bert_model, max_length, batch_size, data_dir):
        super().__init__()
        self.data_path = data_dir
        self.batch_size = batch_size
        self.bert_model = bert_model
        self.max_length = max_length
        self.label_dic = Vocab(os.path.join(data_dir, 'tag.dic'), 'O')
        self.setup()

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        self.train_ds = BertDataSet(os.path.join(self.data_path, 'train.txt'), self.label_dic, self.max_length, tokenizer)
        self.valid_ds = BertDataSet(os.path.join(self.data_path, 'dev.txt'), self.label_dic, self.max_length, tokenizer)
        self.test_ds = BertDataSet(os.path.join(self.data_path, 'test.txt'), self.label_dic, self.max_length, tokenizer)
        
    def train_dataloader(self):
        return data.DataLoader(self.train_ds, self.batch_size, drop_last=True, num_workers=8)

    def test_dataloader(self):
        return data.DataLoader(self.test_ds, self.batch_size, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return data.DataLoader(self.valid_ds, self.batch_size, drop_last=True, num_workers=8)

if __name__ == '__main__':
    dm = NERDataModule(bert_model="hfl/chinese-roberta-wwm-ext", batch_size=10, max_length=128)
        
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-03)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=10, help='批次数据大小')
    parser.add_argument("--pre_train_path", type=str, default="hfl/chinese-roberta-wwm-ext", help='bert模型存储位置')

    param = parser.parse_args()
    param.label_dic = dm.label_dic
    param.tag_size = len(dm.label_dic)

    from model import BERTBiLSTMCRF
    model = BERTBiLSTMCRF(param)
    
    for i, data in enumerate(dm.train_dataloader()):
        print("---------------------", i)
        input_ids, attention_mask, segment_ids, label_ids = data
        import utils
        utils.format_s((input_ids, attention_mask, segment_ids, label_ids)) 

        aa = model.forward(input_ids, attention_mask, segment_ids, label_ids)
        print(aa)
        break