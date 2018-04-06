import argparse

import torch
from torch.autograd import Variable

from RNNPredictNet import RNNPredictNet
from RNNSynthesisNet import RNNSynthesisNet
from utils import *

USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='synthesis',
                        help='prediction/synthesis')
parser.add_argument('--filename', type=str, default='sample',
                    help='filename of .svg file to output, without .svg')
parser.add_argument('--sample_length', type=int, default=800,
                    help='number of strokes to sample')
parser.add_argument(
    '--scale_factor',
    type=int,
    default=10,
    help='factor to scale down by for svg output.  smaller means bigger output')
parser.add_argument('--model_dir', type=str, default='save',
                    help='directory to save model to')

sample_args = parser.parse_args()

with open(os.path.join(sample_args.model_dir, 'config_' + sample_args.type + '.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

saved_args.batch_size = 1

if saved_args.type == 'prediction':
    model = RNNPredictNet(saved_args)
elif saved_args.type == 'synthesis':
    model = RNNSynthesisNet(saved_args)
else:
    print('Unknown model!')

model.load_state_dict(torch.load(sample_args.model_dir + '/model_' + saved_args.type + '.pth')['model'])
model.eval()
if USE_CUDA:
    model = model.cuda()

data_loader = DataLoader(saved_args.batch_size, saved_args.seq_length, saved_args.data_scale,
                         chars=saved_args.chars, points_per_char=saved_args.points_per_char)
chars = 'a quick brown fox jumps over the lazy dog'
# str = 'aaaaabbbbbccccc'
saved_args.U = len(chars)
saved_args.c_dimension = len(data_loader.chars) + 1

str_vec = vectorization(chars, data_loader.char_to_indices)
if USE_CUDA:
    str_vec = Variable(torch.from_numpy(np.array(str_vec).astype(np.float32)[np.newaxis, :, :]).cuda())
else:
    str_vec = Variable(torch.from_numpy(np.array(str_vec).astype(np.float32)[np.newaxis, :, :]))


def sample_stroke():
    [strokes, params] = model.sample(sample_args.sample_length, str_vec)
    draw_strokes(
        strokes,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename + '.normal.svg')
    # draw_strokes_random_color(
    #     strokes,
    #     factor=sample_args.scale_factor,
    #     svg_filename=sample_args.filename + '.color.svg')
    # draw_strokes_random_color(
    #     strokes,
    #     factor=sample_args.scale_factor,
    #     per_stroke_mode=False,
    #     svg_filename=sample_args.filename + '.multi_color.svg')
    # draw_strokes_eos_weighted(
    #     strokes,
    #     params,
    #     factor=sample_args.scale_factor,
    #     svg_filename=sample_args.filename + '.eos_pdf.svg')
    # draw_strokes_pdf(
    #     strokes,
    #     params,
    #     factor=sample_args.scale_factor,
    #     svg_filename=sample_args.filename + '.pdf.svg')
    return [strokes, params]


if __name__ == '__main__':
    sample_stroke()
