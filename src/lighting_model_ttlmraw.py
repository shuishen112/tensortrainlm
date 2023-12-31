import math
from typing import List

import pytorch_lightning as pl
import torch
import torch.jit as jit
from torch import Tensor, nn, optim

from lm_config import args

#  This is the generalization TTLM, the original TTLM


class TensorLayer(jit.ScriptModule):
    """Tensor layer for the tensor network

    Args:
        jit (_type_): _description_
    """

    def __init__(self, cell, *cell_args):
        super(TensorLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        inputs = input.unbind(1)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state]
        return torch.stack(outputs, 1), state


class TTLMCell(jit.ScriptModule):
    """ttlm cell
    Args:
        jit (_type_): _description_
    """

    def __init__(self, rank):
        super(TTLMCell, self).__init__()
        self.rank = rank
        self.activation = nn.Tanh()

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):

        batch_size = input.size(0)

        w2 = input.view(batch_size, self.rank, self.rank)
        # hidden = self.activation(
        #     torch.einsum("bij,bjk->bik", [w1.unsqueeze(1), w2])
        # )  # [batch, 1, rank]
        hidden = torch.einsum("bij,bjk->bik", [state.unsqueeze(1), w2])
        # print(hidden)
        return hidden.squeeze(1)


class TTLMRAWLightningModule(pl.LightningModule):
    """Tensor module"""

    def __init__(self, vocab_size, rank, dropout, lr, cell):
        super().__init__()

        self.vocab_size = vocab_size

        self.rank = rank
        self.dropout = dropout
        self.lr = lr
        self.cell = cell

        # There are two embedding
        # embedding
        self.G = nn.Embedding(self.vocab_size, self.rank * self.rank)
        self.Gt = nn.Linear(self.rank, self.vocab_size)

        nn.init.uniform_(self.G.weight, -0.1, 0.1)

        if cell == "TTLM":
            print("cell_name", cell)
            self.tnn = TensorLayer(TTLMCell, self.rank)
        elif cell == "TinyTNLM2":
            self.tnn = TensorLayer(TTLMCell, self.rank)

        # loss funciton
        self.loss = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(self.dropout)
        self.save_hyperparameters()

    def forward(self, data, hidden):
        embedding = self.dropout(self.G(data))
        output, hidden = self.tnn(embedding, hidden)
        output = self.Gt(output)
        return output.view(-1, self.vocab_size), hidden

    def configure_optimizers(self):
        if args.optim == "adam":
            return optim.Adam(self.parameters(), lr=self.lr)
        elif args.optim == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        batch_size = x.size(0)

        hidden = torch.zeros(batch_size, self.rank).to(self.device)

        output, hidden = self.forward(x, hidden)
        loss = self.loss(output, y)
        perplexity = math.exp(loss.item())

        tensorboard_logs = {
            "perplexity": {"train": perplexity},
            "loss": {"train": loss.detach()},
        }
        self.log(
            "loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "perplexity/train",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        batch_size = x.size(0)

        hidden = torch.zeros(batch_size, self.rank).to(self.device)

        output, hidden = self.forward(x, hidden)
        loss = self.loss(output, y)
        perplexity = math.exp(loss.item())

        tensorboard_logs = {
            "perplexity": {"valid": perplexity},
            "loss": {"valid": loss.detach()},
        }
        self.log(
            "loss_valid", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "perplexity_valid",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        batch_size = x.size(0)
        hidden = torch.zeros(batch_size, self.rank).to(self.device)
        output, hidden = self.forward(x, hidden)
        loss = self.loss(output, y)
        perplexity = math.exp(loss.item())

        tensorboard_logs = {
            "perplexity": {"test": perplexity},
            "loss": {"test": loss.detach()},
        }
        self.log(
            "loss_test", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "perplexity_test",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "log": tensorboard_logs}
