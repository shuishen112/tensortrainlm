import argparse
import random

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_module import Vocab
from data_prep import Prep
from lighting_model_rnn import TextDateModule, TextLightningModule
from lighting_model_tensor import TensorLightningModule
from lm_config import args


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


if __name__ == "__main__":

    # fix the seed
    set_seed()
    p = Prep()
    # Prepare vocab
    train_corpus = p.tokenize(p.train)
    p.building_vocab(train_corpus)

    valid_corpus = p.tokenize(p.valid)
    p.building_vocab(valid_corpus)

    test_corpus = p.tokenize(p.test)
    p.building_vocab(test_corpus)

    word_freqs = p.word_freqs

    print(len(word_freqs))

    # 30 time steps.
    train = Vocab(word_freqs, train_corpus, 31)
    print("len train sample", len(train.encoded_list))

    valid = Vocab(word_freqs, valid_corpus, 31)
    test = Vocab(word_freqs, test_corpus, 31)

    # Train RNN
    vocab_size = len(word_freqs)
    data_module = TextDateModule(train, valid, test, batch_size=20)

    if args.cell in ["RNN", "MRNN", "MIRNN", "Second", "RACs"]:

        model = TextLightningModule(
            vocab_size,
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            dropout=args.dropout,
            lr=args.lr,
            cell=args.cell,
        )
    elif args.cell in ["TinyTNLM", "TinyTNLM2"]:
        model = TensorLightningModule(
            vocab_size=vocab_size,
            rank=args.rank,
            dropout=args.dropout,
            lr=args.lr,
            cell=args.cell,
        )

    # model = model.load_from_checkpoint(
    #     "lightning_logs/tnlm/version_1/checkpoints/epoch=49-step=78950.ckpt"
    # )

    tb_logger = pl_loggers.TensorBoardLogger(
        "./lightning_logs/", name=f"{args.cell}_{args.data_name}"
    )
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=f"{args.cell}_{args.data_name}",
        config=args,
        anonymous=True,
    )
    # Define your gpu here

    checkpoint_callback = ModelCheckpoint(
        monitor="loss_valid",
        dirpath=f"output/{args.cell}_{args.data_name}",
        save_top_k=1,
        filename="sample-{epoch:02d}-{loss_valid:.2f}",
        mode="min",
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=1,
        gpus=1,
        limit_train_batches=10,
        callbacks=checkpoint_callback,
        # resume_from_checkpoint="output/RNN/sample-epoch=30-loss_valid=4.72.ckpt",
        gradient_clip_val=args.clip,
    )
    trainer.fit(
        model,
        data_module,
    )

    result = trainer.test(model, data_module, ckpt_path="best")
    print(result)
