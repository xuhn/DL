# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import collections
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import rnn as rnn_utils
import torch
import torchmetrics
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from torchcrf import CRF
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pytorch_lightning.strategies import DDPStrategy

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

BATCH_SIZE = 64
WORD_VOCAB = '/home/huan/Documents/data/ner_data/word.dic'
LABEL_VOCAB = '/home/huan/Documents/data/ner_data/tag.dic'

# vocab.py
import collections
class Vocab: 
    """文本词表"""
    # 从dict_path导入字典
    def __init__(self, dict_path, unk=None):
        self.unk_token = unk
        self.idx_to_token = ['<pad>']
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        for line in open(dict_path, 'r', encoding='utf-8'):
            token = line.strip('\n')
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        oov_id = self.token_to_idx.get(self.unk_token) if self.unk_token else 0
        return oov_id

class MyData(Dataset):
    def __init__(self, data_path):
        self.word_vocab = Vocab(WORD_VOCAB, 'OOV')
        self.label_vocab = Vocab(LABEL_VOCAB, 'O')
        self.features, self.valid_size, self.labels = self.read(data_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.valid_size[idx], self.labels[idx]

    def read(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)
            features = []
            valid_size = []
            labels = []
            for line in fp.readlines():
                words, names = line.strip('\n').split('\t')
                words = words.split('\002')
                l = [self.word_vocab[token] for token in words]
                valid_size.append(len(l))
                features.append(torch.tensor(l))  

                names = names.split('\002')
                l = [self.label_vocab[token] for token in names]
                labels.append(torch.tensor(l))
            # 填充<pad>
            features = rnn_utils.pad_sequence(features, batch_first=True, padding_value=self.word_vocab['<pad>'])
            labels = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=self.label_vocab['<pad>'])
        return features, valid_size, labels

class NERClassifier(pl.LightningModule):
    def __init__(self, emb_size, hidden_size):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()

        self.word_vocab = Vocab(WORD_VOCAB, 'OOV')
        self.label_vocab = Vocab(LABEL_VOCAB, 'O')
        self.word_emb = nn.Embedding(len(self.word_vocab), emb_size)
        self.gru = nn.GRU(emb_size,
                          hidden_size,
                          num_layers=2,
                          bidirectional=True,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, len(self.label_vocab))
        self.crf = CRF(num_tags=len(self.label_vocab), batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def train_dataloader(self):
        dataset = MyData('/home/huan/Documents/data/ner_data/train.txt')
        return data.DataLoader(dataset, BATCH_SIZE, drop_last=True, num_workers=4)

    def val_dataloader(self):
        dataset = MyData('/home/huan/Documents/data/ner_data/dev.txt')
        return data.DataLoader(dataset, BATCH_SIZE, drop_last=True, num_workers=4)

    def test_dataloader(self):
        dataset = MyData('/home/huan/Documents/data/ner_data/test.txt')
        return data.DataLoader(dataset, BATCH_SIZE, drop_last=True, num_workers=4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def _forward_before_crf(self, inputs, lens):
        inputs = self.dropout(self.word_emb(inputs))
        embed_input_packed = rnn_utils.pack_padded_sequence(inputs, lens.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(embed_input_packed)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        feats = self.fc(output)
        return feats

    def _loss(self, feats, labels):
        mask = labels != self.label_vocab['<pad>']
        loss = self.crf(feats, labels, mask=mask, reduction='mean')
        return -loss


    def _remove_pad_for_labels(self, labels, lens):
        labels_packed = rnn_utils.pack_padded_sequence(labels, lens.cpu(), batch_first=True, enforce_sorted=False)
        labels, _ = rnn_utils.pad_packed_sequence(labels_packed, batch_first=True)
        return labels


    def training_step(self, batch, batch_idx):
        """
        模型训练的前向传播过程
        :param batch:批次数据
        :param batch_idx:
        :param optimizer_idx:
        :return:
        """  
        X, xlen, y = batch
        y = self._remove_pad_for_labels(y, xlen)
        # 1 forward
        # 2 compute the objective function
        feats = self._forward_before_crf(X, xlen)
        J = self._loss(feats, y)
        self.log_dict({'train_avg_loss': J, "step": self.current_epoch}, on_step=False, on_epoch=True)
        return {'loss': J}

    
    def validation_step(self, batch, batch_idx):
        """
        开发集数据验证过程
        :param batch: 批次数据
        :param batch_idx:
        :return:
        """
        X, xlen, y = batch
        y = self._remove_pad_for_labels(y, xlen)
        feats = self._forward_before_crf(X, xlen)
        loss = self._loss(feats, y)
        self.log_dict({'val_avg_loss': loss, "step": self.current_epoch}, on_step=False, on_epoch=True)

        results = self.crf.decode(feats)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result, device=self.device))
        pred = torch.stack(result_tensor)
        
        return {"sentence_lengths": xlen, 'sentence': X, "target": y,
                "pred": pred, "loss": loss}

    def validation_epoch_end(self, outputs):
        """
        验证数据集
        :param outputs: 所有batch预测结果 validation_step的返回值构成的一个list
        :return:
        """
        return self._decode_epoch_end(outputs)

    def _decode_epoch_end(self, outputs):
        """
        对批次预测的结果进行整理，评估对应的结果
        :return:
        """
        gold_list, pred_list = [], []  # 原始标签以及模型预测结果

        for batch_result in outputs:
            batch_size = batch_result['sentence_lengths'].shape[0]
            for i in range(batch_size):
                res = []  # char gold pred
                sentence_gold, sentence_pred = [], []
                for j in range(batch_result['sentence_lengths'][i].item()):
                    char = self.word_vocab.to_tokens([batch_result['sentence'][i][j]])
                    gold = self.label_vocab.to_tokens(batch_result['target'][i][j].item())
                    pred = self.label_vocab.to_tokens(batch_result['pred'][i][j].item())
                    if gold == "<pad>":
                        break
                    sentence_gold.append(gold)
                    sentence_pred.append(pred)
                gold_list.append(sentence_gold)
                pred_list.append(sentence_pred)

        f1 = torch.tensor(f1_score(gold_list, pred_list))
        
        self.log_dict({'val_f1': f1, "step": self.current_epoch})
        return {"val_f1": f1}


    def forward(self, inputs, lens):
        """
        模型实际预测函数
        :param sentences_idx:
        :return:
        """
        feats = self._forward_before_crf(inputs, lens)
        results = self.crf.decode(feats)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result, device=self.device))
        return torch.stack(result_tensor)




if __name__ == '__main__': 

    # saves checkpoints to 'my/path/' at every epoch
    checkpoint_callback = ModelCheckpoint(monitor='val_f1', filename='bi_gru_crf-{epoch:03d}-{val_avg_loss:.2f}-{val_f1:.3f}', mode='max')
    net = NERClassifier(300, 300)
    
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=10, callbacks=[checkpoint_callback], accelerator="gpu", devices=2, strategy=DDPStrategy(find_unused_parameters=False))
    trainer.fit(net)
