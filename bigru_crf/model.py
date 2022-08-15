from argparse import ArgumentParser
from typing import Union, Dict, List, Optional

from pytorch_lightning.utilities.types import STEP_OUTPUT
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch import nn, Tensor
from torchcrf import CRF
from torch.nn.utils import rnn as rnn_utils


class BiLSTMCRF(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-03)
        parser.add_argument('--hidden_dim', type=int, default=300)
        parser.add_argument('--data_path', type=str, default="/home/huan/Documents/data/ResumeNER")
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--char_embedding_size", type=int, default=300)
        parser.add_argument("--experiment", type=bool, default=False)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hyper_params = hparams
        self.lr = hparams.lr
        self.hidden_size = self.hyper_params.hidden_dim
        self.num_layers = 2
        self.bi_directions = True
        self.num_directions = 2 if self.bi_directions else 1
        self.word_emb = nn.Embedding(self.hyper_params.vocab_size, self.hyper_params.char_embedding_size)

        self.lstm = nn.LSTM(input_size=self.hyper_params.char_embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bi_directions)
        self.fc = nn.Linear(self.hyper_params.hidden_dim * 2,
                               self.hyper_params.tag_size)
        self.id2char = self.hyper_params.id2char
        self.idx2tag = self.hyper_params.idx2tag
        self.crf = CRF(num_tags=self.hyper_params.tag_size, batch_first=True)
        self.dropout = nn.Dropout(self.hyper_params.dropout)
        self.hidden_state = self._init_hidden(self.hyper_params.batch_size)

    def configure_optimizers(self):
        """
        配置优化器
        :return:
        """
        optimizer = Adam(self.parameters(),
                          lr=self.lr,
                          weight_decay=self.hyper_params.weight_decay)
        return optimizer

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers * self.num_layers, batch_size, self.hidden_size, device=self.device),
                torch.zeros(self.num_layers * self.num_layers, batch_size, self.hidden_size, device=self.device))

    def _get_batch_info(self, batch):
        this_batch_size = batch.word[0].size()[0]
        sentences_idx = batch.word[0].view(this_batch_size, -1)
        tags = batch.tag[0].view(this_batch_size, -1)
        sentences_length = batch.word[1]
        return sentences_idx, tags, sentences_length

    def _loss(self, feats, labels):
        mask = labels != 1
        loss = self.crf(feats, labels, mask=mask, reduction='mean')
        return -loss

    def _forward_before_crf(self, inputs, lens):
        inputs = self.dropout(self.word_emb(inputs))
        embed_input_packed = rnn_utils.pack_padded_sequence(inputs, lens.cpu(), batch_first=True)
        
        output, _ = self.lstm(embed_input_packed)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        feats = self.fc(output)
        return feats

    
    def training_step(self, batch, batch_idx):
        """
        模型训练的前向传播过程
        :param batch:批次数据
        :param batch_idx:
        :param optimizer_idx:
        :return:
        """  
        X, y, xlen = batch
        # 1 forward
        # 2 compute the objective function
        feats = self._forward_before_crf(X, xlen)
        J = self._loss(feats, y)
        self.log_dict({'train_avg_loss': J, "step": self.current_epoch}, on_step=False, on_epoch=True)
        return {'loss': J}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log_dict({'train_epoch_end_avg_loss': avg_loss, "step": self.current_epoch})
        
        
        
    def validation_step(self, batch, batch_idx):
        """
        开发集数据验证过程
        :param batch: 批次数据
        :param batch_idx:
        :return:
        """
        X, y, xlen = batch
        feats = self._forward_before_crf(X, xlen)
        loss = self._loss(feats, y)
        loss = loss.mean()

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



    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        """
        程序测试模块
        :param batch:
        :param batch_idx:
        :return:
        """
        X, y, xlen = batch
        feats = self._forward_before_crf(X, xlen)
        loss = self._loss(feats, y)
        loss = loss.mean()

        results = self.crf.decode(feats)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result, device=self.device))
        pred = torch.stack(result_tensor)
        
        return {"sentence_lengths": xlen, 'sentence': X, "target": y,
                "pred": pred, "loss": loss}
        

    def test_epoch_end(self, outputs):
        """
        测试集的评估
        :param outputs:测试集batch预测完成结果
        :return:
        """
        return self._decode_epoch_end(outputs)



    def _decode_epoch_end(self, outputs):
        """
        对批次预测的结果进行整理，评估对应的结果
        :return:
        """
        ner_results = []
        gold_list, pred_list = [], []  # 原始标签以及模型预测结果
        g_loss = 0.
        for batch_result in outputs:
            g_loss += batch_result['loss']
            batch_size = batch_result['sentence_lengths'].shape[0]
            for i in range(batch_size):
                res = []  # char gold pred
                sentence_gold, sentence_pred = [], []
                for j in range(batch_result['sentence_lengths'][i].item()):
                    char = self.id2char[batch_result['sentence'][i][j]]
                    gold = self.idx2tag.get(batch_result['target'][i][j].item())
                    pred = self.idx2tag.get(batch_result['pred'][i][j].item())
                    if gold == "<pad>":
                        break
                    res.append(" ".join([char, gold, pred]))
                    sentence_gold.append(gold)
                    sentence_pred.append(pred)
                ner_results.append(res)
                gold_list.append(sentence_gold)
                pred_list.append(sentence_pred)

        f1 = torch.tensor(f1_score(gold_list, pred_list))
        
        self.log_dict({'val_avg_loss': g_loss / len(outputs), 'val_f1': f1, "step": self.current_epoch})
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
