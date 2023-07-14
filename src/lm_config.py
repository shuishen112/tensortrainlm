import argparse

parser = argparse.ArgumentParser()


parser.add_argument(
    "--embedding_size",
    default=100,
    type=int,
    help="embedding size",
)
parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="batch_size",
)
parser.add_argument(
    "--epoch",
    default=50,
    type=int,
    help="epoch",
)
parser.add_argument(
    "--hidden_size",
    default=100,
    type=int,
    help="hidden_size",
)


parser.add_argument(
    "--rank",
    default=10,
    type=int,
    help="rank of tensor train",
)

# parser.add_argument(
#     "--activation",
#     default="nn.Tanh",
#     help="the activation in TNLM[nn.LeakyReLU,nn.RReLU,nn.ReLU,nn.ReLU6,nn.SELU,nn.GELU]",
# )


parser.add_argument(
    "--dropout",
    default=0.25,
    type=float,
    help="dropout",
)

parser.add_argument(
    "--lr",
    default=5e-1,
    type=float,
    help="learning rate",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for initialization",
)

parser.add_argument(
    "--data_name",
    default="ptb",
    help="data name",
)

parser.add_argument(
    "--cell",
    default="RNN",
    help=" cell name: Second, TinyTNLM, RNN, MRNN, MIRNN,RACs, TinyTNLM2",
)

parser.add_argument(
    "--clip",
    default=None,
    type=float,
    help="clip value",
)
parser.add_argument(
    "--project_name",
    default="anonymous",
    help="project name",
)

parser.add_argument(
    "--optim",
    default="sgd",
    help="optim method",
)
args = parser.parse_args()
print(args)
