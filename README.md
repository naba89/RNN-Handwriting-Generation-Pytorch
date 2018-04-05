# Implementation of RNN Handwriting Generation in Pytorch

This repository contains my attempt at implementing Alex Graves wonderful *[paper](https://arxiv.org/abs/1308.0850)* on sequence generation in Pytorch. 

Most of the data pre-processing and sampling code is from [hardmaru's implementation](https://github.com/hardmaru/write-rnn-tensorflow). Since these parts are independent of the framework, can be used directly.

Tried to keep the model and loss function implementation simple to follow. Hope it helps anybody looking for a simple implementation.

## Dependencies
> conda install -c omnia svgwrite\
> sudo apt-get install libmagickwand-dev\
> pip install Wand\
> pip install tensorboardX\
> pytorch >= 0.3

## Data
Before running `train.py` and `sample.py` you need to follow [the instruction](https://github.com/hardmaru/write-rnn-tensorflow#training) and download the necessary files.

You can find a model trained with the default parameters for 30 epochs in the [save](https://github.com/naba89/RNN-Handwriting-Generation-Pytorch/tree/master/save) directory. Below is the training curve for the trained model.

![example](https://rawgit.com/naba89/RNN-Handwriting-Generation-Pytorch/master/loss.png)

A generated sample from the trained network:


![example](https://rawgit.com/naba89/RNN-Handwriting-Generation-Pytorch/master/output/sample.color.svg)

Well it does look like handwriting! 

##### Coming soon: Text-to-Handwriting
