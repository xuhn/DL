from argparse import ArgumentParser

from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch import nn, Tensor
from torchcrf import CRF
from torch.nn.utils import rnn as rnn_utils
from dataloader import NERDataModule
from transformers import BertModel
import utils

class BERTBiLSTMCRF(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-04)
        parser.add_argument('--hidden_dim', type=int, default=300)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hyper_params = hparams
        self.lr = hparams.lr
        self.hidden_size = self.hyper_params.hidden_dim
        self.num_layers = 2
        self.bi_directions = True
        self.num_directions = 2 if self.bi_directions else 1
        
        self.label_dic = self.hyper_params.label_dic
        
        self.bert = BertModel.from_pretrained(hparams.pre_train_path)

        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bi_directions,
                            batch_first=True)
        self.fc = nn.Linear(self.hyper_params.hidden_dim * 2,
                               self.hyper_params.tag_size)
        
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
        mask = labels != utils.IGNORE_LABEL
        loss = self.crf(feats, labels, mask=mask, reduction='mean')
        return -loss

    def _forward_before_crf(self, input_ids, attention_mask, segment_ids):
        embeds = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
        output, _ = self.lstm(embeds)
        output = self.dropout(output)
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
        input_ids, attention_mask, segment_ids, label_ids = batch
        # 1 forward
        # 2 compute the objective function
        feats = self._forward_before_crf(input_ids, attention_mask, segment_ids)
        loss = self._loss(feats, label_ids)
        self.log_dict({'train_avg_loss': loss, "step": self.current_epoch}, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        开发集数据验证过程
        :param batch: 批次数据
        :param batch_idx:
        :return:
        """
        input_ids, attention_mask, segment_ids, label_ids = batch
        feats = self._forward_before_crf(input_ids, attention_mask, segment_ids)
        loss = self._loss(feats, label_ids)
        self.log_dict({'val_avg_loss': loss, "step": self.current_epoch}, on_step=False, on_epoch=True, sync_dist=True)
        
        results = self.crf.decode(feats)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result, device=self.device))
        pred = torch.stack(result_tensor)
        
        return {"target": label_ids, "pred": pred, "loss": loss}

    def validation_epoch_end(self, outputs):
        """
        验证数据集
        :param outputs: 所有batch预测结果 validation_step的返回值构成的一个list
        :return:
        """
        return self._decode_epoch_end(outputs)


    def test_step(self, batch, batch_idx):
        """
        程序测试模块
        :param batch:
        :param batch_idx:
        :return:
        """
        input_ids, attention_mask, segment_ids, label_ids = batch
        feats = self._forward_before_crf(input_ids, attention_mask, segment_ids)
        loss = self._loss(feats, label_ids)
        self.log_dict({'test_avg_loss': loss, "step": self.current_epoch}, on_step=False, on_epoch=True, sync_dist=True)
        
        results = self.crf.decode(feats)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result, device=self.device))
        pred = torch.stack(result_tensor)
        
        return {"target": label_ids, "pred": pred, "loss": loss}
        

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
        gold_list, pred_list = [], []  # 原始标签以及模型预测结果
        for batch_result in outputs:
            batch_size = batch_result['target'].shape[0]
            for i in range(batch_size):
                sentence_gold, sentence_pred = [], []
                for j in range(1, len(batch_result['target'][i])):
                    if batch_result['target'][i][j].item() == utils.IGNORE_LABEL:
                        break;

                    gold = self.label_dic.to_tokens(batch_result['target'][i][j].item())
                    pred = self.label_dic.to_tokens(batch_result['pred'][i][j].item())
                    sentence_gold.append(gold)
                    sentence_pred.append(pred)
                gold_list.append(sentence_gold)
                pred_list.append(sentence_pred)
                
        f1 = torch.tensor(f1_score(gold_list, pred_list))
        
        self.log_dict({'val_f1': f1, "step": self.current_epoch})
        return {"val_f1": f1}


    def forward(self, input_ids, attention_mask, segment_ids, label_ids=None):
        """
        模型实际预测函数
        :param sentences_idx:
        :return:
        """
        feats = self._forward_before_crf(input_ids, attention_mask, segment_ids)
        
        if label_ids is not None:
            loss = self._loss(feats, label_ids)
            return loss
        else:
            results = self.crf.decode(feats)
            result_tensor = []
            for result in results:
                result_tensor.append(torch.tensor(result, device=self.device))
            return torch.stack(result_tensor)

if __name__ == '__main__':
    dm = NERDataModule(bert_model="hfl/chinese-roberta-wwm-ext", batch_size=2)
    aa, bb, cc, dd = next(iter(dm.train_dataloader()))
    print(aa)
    