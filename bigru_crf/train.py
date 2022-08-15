import json
import os
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer
from model import BiLSTMCRF
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from dataloader import NERDataModule
from pytorch_lightning.strategies import DDPStrategy

pl.seed_everything(2022)


def train(args):
    path_prefix = "model_save"
    os.makedirs(path_prefix, exist_ok=True)

    ner_dm = NERDataModule(data_dir=args.data_path, batch_size=args.batch_size)
    args.tag_size = ner_dm.tag_size
    args.vocab_size = ner_dm.vocab_size
    args.id2char = ner_dm.id2char
    args.idx2tag = ner_dm.idx2tag
    if args.load_pre:
        model = BiLSTMCRF.load_from_checkpoint(args.ckpt_path, hparams=args)
    else:
        model = BiLSTMCRF(args)
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(save_top_k=3,
                                          monitor="val_f1",
                                          mode="max",
                                          dirpath=path_prefix,
                                          filename="ner-{epoch:03d}-{val_f1:.3f}", )
    trainer = Trainer.from_argparse_args(args, callbacks=[lr_logger,
                                                          checkpoint_callback],
                                         accelerator="gpu", devices=2, strategy=DDPStrategy(find_unused_parameters=False),
                                         max_epochs=200)

    if args.train:
        trainer.fit(model=model, datamodule=ner_dm)

    if args.test:
        trainer.test(model, ner_dm)

    if args.save_state_dict:
        if len(os.name) > 0:
            ner_dm.save_dict(path_prefix)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--load_pre", default=False, action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="model_save/ner-epoch=388-val_f1=0.924.ckpt")
    parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--save_state_dict", default=True, action="store_true")
    parser = BiLSTMCRF.add_model_specific_args(parser)
    params = parser.parse_args()
    train(params)
