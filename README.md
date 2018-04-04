# Implementation of RNN Handwriting Generation in Pytorch

This repository contains my attempt at implementing Alex Graves wonderful *[paper](https://arxiv.org/abs/1308.0850)* on sequence generation in Pytorch. 

Most of the data pre-processing and sampling code is from [hardmaru's implementation](https://github.com/hardmaru/write-rnn-tensorflow). Since these parts are independent of the framework, can be used directly.

Tried to keep the model and loss function implementation simple to follow. Hope it helps anybody looking for a simple implementation.

## Data
Before running `train.py` and `sample.py` you need to follow [the instruction](https://github.com/hardmaru/write-rnn-tensorflow#training) and download the necessary files.

I will upload the some results, training curves and trained models shortly.
