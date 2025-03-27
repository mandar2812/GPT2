# GPT-2 PyTorch Implementation

![build](https://github.com/mandar2812/GPT2/workflows/build/badge.svg)
![GitHub](https://img.shields.io/github/license/mandar2812/GPT2)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/affjljoo3581/GPT2/blob/master/GPT2_Interactive_Notebook.ipynb)
[![codecov](https://codecov.io/gh/mandar2812/GPT2/branch/master/graph/badge.svg)](https://codecov.io/gh/affjljoo3581/GPT2)

[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Table of contents

* [Introduction](#introduction)
* [Dependencies](#dependencies)
* [Usage](#usage)
  * [How to train?](#how-to-train)
  * [Generate sentences!](#generate-sentences)
  * [Evaluate the model](#evaluate-the-model)
  * [Visualize metrics](#visualize-metrics)
* [Using apex in training](#using-apex-in-training)
* [Play in Google Colab!](#play-in-google-colab)
* [License](#license)


## Introduction
This is a fork of the [affjljoo3581](https://github.com/affjljoo3581)'s excellent [GPT2 implementation](https://github.com/affjljoo3581/GPT2) implementation, with QA fine-tuning and rudimentary chat capabilities.

## Dependencies
* regex
* tqdm
* torch
* numpy
* matplotlib
* pandas
* Hugging Face Datasets
* Hugging Face Tokenizers

## Usage

### How to (pre) train?

Before pre-training GPT-2, corpus dataset should be prepared. We recommend to build your own corpus by using [Expanda](https://github.com/mandar2812/Expanda).

After preparing datasets, you can pre-train GPT-2 by using as follows:

    $ pipenv run python -m src.gpt2 train DATA_CORPUS_BUILD_DIR \
        --train_corpus           corpus.train.txt \
        --eval_corpus            corpus.test.txt \
        --tokenizer_path         tokenizer.json \
        --save_checkpoint_path   ckpt-gpt2.pth \
        --save_model_path        gpt2-pretrained.pth \
        --batch_train            32 \
        --batch_eval             32 \
        --seq_len                256 \
        --total_steps            10000 \
        --eval_steps             1000 \
        --save_steps             1000 \
        --layers                 12 \
        --use_amp --use_grad_ckpt

To resume training from last checkpoint file, use `--from_checkpoint [last checkpoint file]` option.
If you want to train GPT-2 with multiple GPUs, use `--gpus [number of gpus]` option.

The detail of command-line usage is as follows:

    usage: gpt2 train [-h] --train_corpus TRAIN_CORPUS --eval_corpus EVAL_CORPUS --tokenizer_path TOKENIZER_PATH [--seq_len SEQ_LEN] [--layers LAYERS] [--heads HEADS] [--dims DIMS] [--rate RATE] [--dropout DROPOUT]
                  [--batch_train BATCH_TRAIN] [--batch_eval BATCH_EVAL] [--base_lr BASE_LR] [--wd_rate WD_RATE] [--total_steps TOTAL_STEPS] [--eval_steps EVAL_STEPS] [--save_steps SAVE_STEPS] [--save_version_steps SAVE_VERSION_STEPS]
                  [--save_model_path SAVE_MODEL_PATH] [--save_checkpoint_path SAVE_CHECKPOINT_PATH] [--from_checkpoint FROM_CHECKPOINT] [--from_pretrained FROM_PRETRAINED] [--use_amp] [--use_grad_ckpt] [--gpus GPUS]
                  corpus_dir

    options:
      -h, --help            show this help message and exit

    Corpus and vocabulary:
      corpus_dir            root directory of corpus files
      --train_corpus TRAIN_CORPUS
                            training corpus file path
      --eval_corpus EVAL_CORPUS
                            evaluation corpus file path
      --tokenizer_path TOKENIZER_PATH
                            tokenizer file path

    Model configurations:
      --seq_len SEQ_LEN     maximum sequence length
      --layers LAYERS       number of transformer layers
      --heads HEADS         number of multi-heads in attention layer
      --dims DIMS           dimension of representation in each layer
      --rate RATE           increase rate of dimensionality in bottleneck
      --dropout DROPOUT     probability that each element is dropped

    Training and evaluation:
      --batch_train BATCH_TRAIN
                            number of training batch size
      --batch_eval BATCH_EVAL
                            number of evaluation batch size
      --base_lr BASE_LR     default learning rate
      --wd_rate WD_RATE     weight decay rate
      --total_steps TOTAL_STEPS
                            number of total training steps
      --eval_steps EVAL_STEPS
                            period to evaluate model and record metrics
      --save_steps SAVE_STEPS
                            period to save training state to checkpoint
      --save_version_steps SAVE_VERSION_STEPS
                            period to save a versioned/branched model.

    Saving and restoring:
      --save_model_path SAVE_MODEL_PATH
                            save trained model weights to the file
      --save_checkpoint_path SAVE_CHECKPOINT_PATH
                            save training state to the checkpoint file
      --from_checkpoint FROM_CHECKPOINT
                            load last training state from checkpoint file
      --from_pretrained FROM_PRETRAINED
                            initialize parameters from pretrained model

    Extensions:
      --use_amp             use automatic mixed-precision in training
      --use_grad_ckpt       use gradient checkpointing in transformer layers
      --gpus GPUS           number of gpu devices to use in training


### Generate sentences

After training GPT-2, you can generate sentences with your trained model in interactive mode.

    $ python -m gpt2 generate --vocab_path      build/vocab.txt \
                              --model_path      model.pth \
                              --seq_len         64 \
                              --nucleus_prob    0.8

The detail of command-line usage is as follows:

    usage: gpt2 generate [-h] --vocab_path VOCAB_PATH --model MODEL
                         [--seq_len SEQ_LEN] [--layers LAYERS] [--heads HEADS]
                         [--dims DIMS] [--rate RATE] [--top_p TOP_P] [--use_gpu]

    optional arguments:
      -h, --help            show this help message and exit
      --vocab_path VOCAB_PATH
                            vocabulary file path
      --model_path MODEL_PATH
                            trained GPT-2 model file path

    Model configurations:
      --seq_len SEQ_LEN     maximum sequence length
      --layers LAYERS       number of transformer layers
      --heads HEADS         number of multi-heads in attention layer
      --dims DIMS           dimension of representation in each layer
      --rate RATE           increase rate of dimensionality in bottleneck

    Generating options:
      --nucleus_prob NUCLEUS_PROB
                            probability threshold for nucleus sampling
      --use_gpu             use gpu device in inferencing

### Evaluate the model

To estimate the performance of trained model, calculate the objective metrics on the evaluation dataset.

    $  pipenv run python -m src.gpt2 evaluate DATA_CORPUS_BUILD_DIR \
        --eval_corpus            corpus.test.txt \
        --tokenizer_path         tokenizer.json \
        --model_path             gpt2-pretrained.pth \
        --batch_eval             96 \               
        --seq_len                256 \                                                         
        --layers                 12 \
        --use_gpu 


### Visualize metrics

You can also analyse training loss graph by visualizing recorded metrics.

    $ pipenv run python -m src.gpt2 visualize \
        --model_path path/to/gpt2-pretrained.pth \
        --figure src/build/gpt2-training.png

The example figure is as bellow:

![figure](./example-figure.png)

## Mixed Precision Training

Mixed precision training is possible with `torch.amp`, provided your GPU meets the requirements. Use the `--use_amp` flag in the training program.

## Play in Google Colab!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/affjljoo3581/GPT2/blob/master/GPT2_Interactive_Notebook.ipynb)

You can play trained GPT2 model in Google Colab! The above notebook contains text generation and metrics evaluation. You need to upload the trained model, vocabulary file and evaluation dataset to [Google Cloud Storage](https://cloud.google.com/storage).

For the people who are interested in korean-version of GPT2, we rewrite the above notebook to provide the case of `gpt2-ko-302M` model especially, which is trained with about **5.04B** tokens from korean documents. You can play demo [in this notebook](https://colab.research.google.com/github/affjljoo3581/GPT2/blob/master/korean_gpt2_302M_demo.ipynb).

## License
This project is [Apache-2.0 Licensed](./LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=affjljoo3581/GPT2&type=Date)](https://star-history.com/#affjljoo3581/GPT2&Date)
