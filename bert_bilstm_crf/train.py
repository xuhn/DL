import json
import os
from argparse import ArgumentParser
from transformers import BertTokenizer
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from model import BERTBiLSTMCRF
from dataloader import NERDataModule
from pytorch_lightning.strategies import DDPStrategy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
pl.seed_everything(2022)

def training(param):
    dm = NERDataModule(batch_size=param.batch_size, bert_model=param.pre_train_path, max_length=param.max_length, data_dir=param.data_dir)
    checkpoint_callback = ModelCheckpoint(monitor='val_f1',
                                          mode="max",
                                          filename="bert-{epoch:03d}-{val_avg_loss:.2f}-{val_f1:.3f}",
                                          dirpath=param.save_dir,
                                          save_top_k=3)

    param.label_dic = dm.label_dic
    param.tag_size = len(dm.label_dic)
    model = BERTBiLSTMCRF(param)
    
    if param.load_pre:
        model = model.load_from_checkpoint(param.pre_ckpt_path, hparams=param)
        
    logger = TensorBoardLogger("log_dir", name="bert_pl")

    trainer = pl.Trainer(logger=logger, 
                         accelerator="gpu", devices=2, strategy=DDPStrategy(find_unused_parameters=True),
                         callbacks=[checkpoint_callback],
                         max_epochs=param.epoch,
                         # precision=16,
                         accumulate_grad_batches=10,  # 由于使用bert时，批次数据量太少，设置多个批次后再进行梯度处理
                         # limit_train_batches=0.1,
                         # limit_val_batches=0.1,
                         # gradient_clip_val=0.5
                         )
    if param.train:
        trainer.fit(model=model, datamodule=dm)
    if param.test:
        trainer.test(model=model, datamodule=dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10, help='批次数据大小')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default="model_save/bert", help='模型存储位置')
    parser.add_argument('--data_dir', type=str, default="/home/huan/Documents/data/ner_data", help='数据存储位置')
    parser.add_argument('--load_pre', type=bool, default=False, action="store_true", help='是否加载已经训练好的ckpt')
    parser.add_argument('--test', type=bool, default=True, action="store_true", help='是否测试数据')
    parser.add_argument('--train', type=bool, default=False, action="store_true", help='是否训练')
    parser.add_argument('--max_length', type=int, default=200, help='截取句子的最大长度')
    parser.add_argument('--pre_ckpt_path', type=str,
                        default="model_save/bert/bert-epoch=018-val_avg_loss=0.96-val_f1=0.994.ckpt",
                        help='是否加载已经训练好的ckpt')
    parser.add_argument("--pre_train_path", type=str, default="hfl/chinese-roberta-wwm-ext", help='bert模型存储位置')
    
    parser = BERTBiLSTMCRF.add_model_specific_args(parser)
    args = parser.parse_args()
    print('args: ', args)
    training(args)

