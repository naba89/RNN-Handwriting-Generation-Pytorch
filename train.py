import argparse
import os
import pickle
import sys
import time

import numpy as np

import torch.optim as optim
import torch
from torch.autograd import Variable

from RNNSynthesisNet import RNNSynthesisNet
from loss_functions import PredictionLoss
from RNNPredictNet import RNNPredictNet
from utils import DataLoader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

USE_CUDA = torch.cuda.is_available()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='synthesis',
                        help='prediction/synthesis')
    parser.add_argument('--input_size', type=int, default=3,
                        help='input num features')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--chars', type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
                        help='chars')
    parser.add_argument('--points_per_char', type=int, default=25,
                        help='points per char (appr.)')
    parser.add_argument('--bidirectional', type=bool, default=False,
                        help='use BLSTM')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--seq_length', type=int, default=300,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--num_mixture', type=int, default=20,
                        help='number of gaussian mixtures')
    parser.add_argument('--data_scale', type=float, default=20,
                        help='factor to scale raw data down by')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--validate_every', type=int, default=10,
                        help='frequency of validation')
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale)
    args.c_dimension = len(data_loader.chars) + 1
    args.U = data_loader.max_U

    if args.model_dir != '' and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    with open(os.path.join(args.model_dir, 'config_' + args.type + '.pkl'), 'wb') as f:
        pickle.dump(args, f)

    if args.type == 'prediction':
        model = RNNPredictNet(args)
    elif args.type == 'synthesis':
        model = RNNSynthesisNet(args)
    else:
        print('Unknown model!')
        sys.exit()
    if USE_CUDA:
        model = model.cuda()

    loss_fn = PredictionLoss(args.batch_size, args.seq_length)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay_rate)

    training_loss = []
    validation_loss = []

    for e in range(args.num_epochs):
        data_loader.reset_batch_pointer()
        # v_x, v_y = data_loader.validation_data()
        # v_x = torch.from_numpy(np.array(v_x))
        # v_y = torch.from_numpy(np.array(v_y))
        # if USE_CUDA:
        #     v_x = v_x.cuda()
        #     v_y = v_y.cuda()

        for b in range(data_loader.num_batches):
            model.train()
            train_step = e * data_loader.num_batches + b
            start = time.time()

            x, y, c_vec, c = data_loader.next_batch()
            x = torch.from_numpy(np.array(x))
            y = torch.from_numpy(np.array(y))
            # print(c_vec)
            c_vec = torch.from_numpy(np.array(c_vec).astype(np.float32))
            if USE_CUDA:
                x = x.cuda()
                y = y.cuda()
                c_vec = c_vec.cuda()

            x = Variable(x)
            y = Variable(y)
            c_vec = Variable(c_vec)

            optimizer.zero_grad()
            if args.type == 'synthesis':
                output = model(x, c_vec)
            else:
                output = model(x)

            train_loss = loss_fn(output, y)

            train_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

            optimizer.step()

            training_loss.append(train_loss.data[0])

            # model.eval()
            # output = model(Variable(v_x, volatile=True))
            # val_loss = loss_fn(output, Variable(v_y, volatile=True))
            # validation_loss.append(val_loss.data[0])

            end = time.time()
            print(
                "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                    train_step,
                    args.num_epochs * data_loader.num_batches,
                    e,
                    train_loss.data[0],
                    end - start))

            if (train_step % args.save_every == 0) and (train_step > 0):
                checkpoint_path = os.path.join(
                    args.model_dir, 'model_' + args.type + '.pth')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': e,
                    'current_lr': args.learning_rate * (args.decay_rate ** e)
                },
                    checkpoint_path)
                from sample import sample_stroke
                sample_stroke()
                print("model saved to {}".format(checkpoint_path))
        lr_scheduler.step()


if __name__ == '__main__':
    main()
