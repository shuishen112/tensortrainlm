<a name="readme-top"></a>

# Language Modeling using Tensor Trains
---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-orange)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## What's This Project About?

![images/TTLM.png](images/TTLM.png)

The project implemented TTLM, Vanilla RNN, RACs, Second-order RNN  language models by using PyTorch default functions.


Hyper-parameters are:

| **Hyper-parameter** |       **Value**       |
|:-------------------:|:---------------------:|
|      batch size     |           20          |
|   sequence length   |           30          |
|   number of layers  |           1           |
|    embedding size   |          400          |
|     hidden size     |          20          |

## Dataset

<p align="right">(<a href="#readme-top">back to top</a>)</p>
The data used are word-level language modeling dataset [wikitext2](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) and [ptb](https://data.deepai.org/ptbdataset.zip)
as presented in `./data` directory.
## Structure of the Project

The structure of the project a]is as follows:

```bash
.
├── data
│   ├── wiki.test.txt
│   ├── wiki.val.txt
    ├── wiki.train.txt
    ├── ptb.test.txt
    ├── ptb.val.txt
│   └── ptb.valid.txt

├── LICENSE
├── README.md
└── src
    ├── data_module.py
    ├── data_prep.py
    ├── lighting_model_rnn.py
    ├── lighting_model_tensor.py
    ├── lighting_model_ttlmraw.py
    ├── lm_config.py
    └── pl_trainer.py
```


## Dependencies
<p align="right">(<a href="#readme-top">back to top</a>)</p>
The following dependencies are required, please install by using the virtual environment
of your choice:

```

pip install torch torchvision torchaudio
pip install pytorch_lightning
pip install nltk
python -m nltk.downloader punkt
pip install tensorboard
pip install wandb
```

## How to Run?

To run the project, use commend `python src/pl_trainer.py --cell=TinyTNLM2 --rank=20 --data_name=ptb`

- tensorboard
To view the result, run the commend `tensorboard --logdir ./lightning_logs/` in
`./src`
- wandb

## Wandb Result

- WikiText-2 

![images/wiki.png](images/wiki.png)

- ptb

![images/ptb.png](images/ptb.png)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
