import argparse

import torch

from model import RNNPredictNet
from utils import *

USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser()
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

with open(os.path.join(sample_args.model_dir, 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

model = RNNPredictNet(saved_args)
model.load_state_dict(torch.load(sample_args.model_dir + '/model.pth')['model'])
model.eval()
if USE_CUDA:
    model = model.cuda()


def sample_stroke():
    [strokes, params] = model.sample(sample_args.sample_length)
    draw_strokes(
        strokes,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename + '.normal.svg')
    draw_strokes_random_color(
        strokes,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename + '.color.svg')
    draw_strokes_random_color(
        strokes,
        factor=sample_args.scale_factor,
        per_stroke_mode=False,
        svg_filename=sample_args.filename + '.multi_color.svg')
    draw_strokes_eos_weighted(
        strokes,
        params,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename + '.eos_pdf.svg')
    draw_strokes_pdf(
        strokes,
        params,
        factor=sample_args.scale_factor,
        svg_filename=sample_args.filename + '.pdf.svg')
    return [strokes, params]


if __name__ == '__main__':
    sample_stroke()
