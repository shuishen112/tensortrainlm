import math
from typing import List

import pytorch_lightning as pl
import torch
import torch.jit as jit
import wandb
from torch import Tensor, nn, optim
from torch.nn import Parameter


class TextDateModule(pl.LightningDataModule):
    """Pytorch lightning data module."""

    def __init__(self, train_corpus, valid_corpus, test_corpus, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train = train_corpus
        self.valid = valid_corpus
        self.test = test_corpus

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, self.batch_size, num_workers=16, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid, self.batch_size, num_workers=16, shuffle=False, drop_last=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test, self.batch_size, num_workers=16, shuffle=False, drop_last=True
        )


class MIRNNCell(jit.ScriptModule):
    """Multiplicative integration cell

    Args:
        jit (_type_): _description_
    """

    def __init__(self, input_size, hidden_size):
        super(MIRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(hidden_size))
        self.bias_hh = Parameter(torch.randn(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        hx = state
        hidden = (torch.mm(input, self.weight_ih.t()) + self.bias_ih) * (
            torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        )

        hy = torch.tanh(hidden)
        return hy


class RACs(jit.ScriptModule):
    """RACs cell

    Args:
        jit (_type_): _description_
    """

    def __init__(self, input_size, hidden_size):
        super(RACs, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(hidden_size))
        self.bias_hh = Parameter(torch.randn(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        hx = state
        hidden = (torch.mm(input, self.weight_ih.t()) + self.bias_ih) * (
            torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        )

        return hidden


class SecondOrderCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(SecondOrderCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.three_order = nn.Bilinear(
            self.input_size, self.hidden_size, self.hidden_size
        )

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        hidden = self.three_order(input, state)
        hy = hidden
        # hy = torch.tanh(hidden)
        return hy


class TensorRNNCell(jit.ScriptModule):
    """Tensor RNN cell

    Args:
        jit (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, input_size, hidden_size):
        super(TensorRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tensor = Parameter(torch.randn(input_size, hidden_size * hidden_size))
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.bias_ih = Parameter(torch.randn(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        w_hh = torch.mm(input, self.tensor).reshape(
            -1, self.hidden_size, self.hidden_size
        )
        hidden = (
            torch.mm(input, self.weight_ih.t())
            + torch.bmm(state.unsqueeze(1), w_hh).squeeze(1)
            + self.bias_ih
        )

        hy = torch.tanh(hidden)
        return hy


class MRNNCell(jit.ScriptModule):
    """
        multiplicative RNN cell
    Args:
        jit (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, input_size, hidden_size):
        super(MRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.Wih = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)

        self.w_im = nn.Linear(input_size, hidden_size, bias=False)
        self.w_hm = nn.Linear(hidden_size, hidden_size, bias=False)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):

        mx = self.w_im(input) * self.w_hm(state)
        wi = self.Wih(input)
        wh = self.Whh(mx)

        hidden = torch.tanh(wi + wh)
        return hidden


class RNNCell(jit.ScriptModule):
    """standard rnn cell

    Args:
        jit (_type_): _description_
    """

    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(hidden_size))
        self.bias_hh = Parameter(torch.randn(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        hx = state
        hidden = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )

        hy = torch.tanh(hidden)
        return hy


class RNNLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor):
        inputs = input.unbind(1)

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state]
        return torch.stack(outputs, 1), state


class TextLightningModule(pl.LightningModule):
    """RNN module"""

    def __init__(self, vocab_size, hidden_size, embedding_size, dropout, lr, cell):
        super().__init__()

        self.hidden_size = hidden_size  # 200
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.lr = lr
        self.cell = cell
        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # layers
        print("cell name", self.cell)
        if self.cell == "MIRNN":
            self.rnn = RNNLayer(MIRNNCell, embedding_size, hidden_size)
        elif self.cell == "MRNN":
            self.rnn = RNNLayer(MRNNCell, embedding_size, hidden_size)
        elif self.cell == "RNN":
            self.rnn = RNNLayer(RNNCell, embedding_size, hidden_size)
        elif self.cell == "RACs":
            self.rnn = RNNLayer(RACs, embedding_size, hidden_size)
        elif self.cell == "Second":
            self.rnn = RNNLayer(SecondOrderCell, embedding_size, hidden_size)
        else:
            print("there is no cell")

        self.out_embed = nn.Linear(self.hidden_size, self.embedding_size)
        self.out_fc = nn.Linear(self.embedding_size, vocab_size)
        # loss funciton

        # ties the weights of output embeddings with the input embeddings
        self.out_fc.weight = self.embedding.weight
        self.loss = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, data, hidden):
        embedding = self.dropout(self.embedding(data))
        output, hidden = self.rnn(embedding, hidden)
        output_embed = self.out_embed(output)
        output = self.out_fc(output_embed)
        return output.view(-1, self.vocab_size), hidden

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        batch_size = x.size(0)

        hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)

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

        hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
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

        hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
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

    def init_hidden(self, batch_size=20):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden
