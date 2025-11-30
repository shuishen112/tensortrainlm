import math
from typing import List

import pytorch_lightning as pl
import torch
import torch.jit as jit
from torch import Tensor, nn, optim

from lm_config import args


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


class TTLMLargeCell(jit.ScriptModule):
    """tnlm cell

    Args:
        jit (_type_): _description_
    """

    def __init__(self, rank):
        super(TTLMLargeCell, self).__init__()
        self.rank = rank
        self.wih = nn.Linear(self.rank, self.rank)
        self.whh = nn.Linear(self.rank * self.rank, self.rank * self.rank)
        self.activation = nn.Tanh()

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):

        batch_size = input.size(0)
        w1 = self.wih(state)
        w2 = self.whh(input).view(batch_size, self.rank, self.rank)
        # w2 = unit.view(batch_size, self.rank, self.rank)
        # hidden = self.activation(
        #     torch.einsum("bij,bjk->bik", [w1.unsqueeze(1), w2])
        # )  # [batch, 1, rank]
        hidden = torch.einsum("bij,bjk->bik", [w1.unsqueeze(1), w2])
        # print(hidden)
        return hidden.squeeze(1)


class TTLMTinyCell(jit.ScriptModule):
    """tnlm cell 2

    Args:
        jit (_type_): _description_
    """

    def __init__(self, rank):
        super(TTLMTinyCell, self).__init__()
        self.rank = rank
        self.wih = nn.Linear(self.rank, self.rank)
        self.activation = nn.Tanh()

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):

        batch_size = input.size(0)
        w1 = self.wih(state)
        w2 = input.view(batch_size, self.rank, self.rank)

        # hidden = self.activation(
        #     torch.einsum("bij,bjk->bik", [w1.unsqueeze(1), w2])
        # )  # [batch, 1, rank]
        hidden = torch.einsum("bij,bjk->bik", [w1.unsqueeze(1), w2])
        return hidden.squeeze(1)


class TensorLightnightModuleClassification(pl.LightningModule):
    """Tensor module for classification"""

    def __init__(self, vocab_size, rank, dropout, lr, cell, num_classes):
        super().__init__()

        self.vocab_size = vocab_size
        self.rank = rank
        self.dropout = dropout
        self.lr = lr
        self.cell = cell
        self.num_classes = num_classes
        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.rank * self.rank)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        if cell == "large":
            print("cell_name", cell)
            self.tnn = TensorLayer(TTLMLargeCell, self.rank)
        elif cell == "tiny":
            self.tnn = TensorLayer(TTLMTinyCell, self.rank)

        self.out_fc = nn.Linear(self.rank, 1)
        self.sigmoid = nn.Sigmoid()

        # loss funciton
        self.loss = nn.BCELoss()

        self.dropout = nn.Dropout(self.dropout)
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, data, hidden):
        embedding = self.dropout(self.embedding(data))
        output, hidden = self.tnn(embedding, hidden)
        output = self.sigmoid(self.out_fc(hidden)).squeeze()
        return output, hidden

    def configure_optimizers(self):
        if args.optim == "adam":
            return optim.Adam(self.parameters(), lr=self.lr)
        elif args.optim == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # y = y.view(-1)
        batch_size = x.size(0)

        hidden = torch.zeros(batch_size, self.rank).to(self.device)

        output, hidden = self.forward(x, hidden)
        loss = self.loss(output, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y = y.view(-1)
        batch_size = x.size(0)
        hidden = torch.zeros(batch_size, self.rank).to(self.device)
        output, hidden = self.forward(x, hidden)

        loss = self.loss(output, y)
        self.log("loss_valid", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss_valid": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # y = y.view(-1)
        batch_size = x.size(0)
        hidden = torch.zeros(batch_size, self.rank).to(self.device)
        output, hidden = self.forward(x, hidden)

        # convert the output to 0 or 1
        output = (output > 0.5).float()
        correct = (output == y).float().sum()
        self.test_step_outputs.append({"correct": correct, "batch_size": batch_size})
        return {"correct": correct}

    def on_test_epoch_end(self):
        correct = sum(output["correct"] for output in self.test_step_outputs)
        total = sum(output["batch_size"] for output in self.test_step_outputs)
        accuracy = correct / total
        self.log("accuracy_test", accuracy)

class TensorLightningModule(pl.LightningModule):
    """Tensor module"""

    def __init__(self, vocab_size, rank, dropout, lr, cell):
        super().__init__()

        self.vocab_size = vocab_size

        self.rank = rank
        self.dropout = dropout
        self.lr = lr
        self.cell = cell
        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.rank * self.rank)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        if cell == "large":
            print("cell_name", cell)
            self.tnn = TensorLayer(TTLMLargeCell, self.rank)
        elif cell == "tiny":
            self.tnn = TensorLayer(TTLMTinyCell, self.rank)

        self.out_embed = nn.Linear(self.rank, self.rank * self.rank)
        self.out_fc = nn.Linear(self.rank * self.rank, vocab_size)
        # self.out_fc = nn.Linear(self.rank, vocab_size)

        self.out_fc.weight = self.embedding.weight

        # loss funciton
        self.loss = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(self.dropout)
        self.save_hyperparameters()

    def forward(self, data, hidden):
        embedding = self.dropout(self.embedding(data))
        output, hidden = self.tnn(embedding, hidden)
        output_embed = self.out_embed(output)
        output = self.out_fc(output_embed)
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
