# for command line
import torch
from training import Optimizing
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-i', '--iteration', type=int, default=500, help='Max iteration for optimization')
parser.add_argument('-a', '--alpha', type=int, default=-2, help='The exponential index of content loss weight')
parser.add_argument('-b', '--beta',  type=int, default=9,  help='The exponential index of style loss weight')
parser.add_argument('-l', '--learnrate', type=float, default=1, help='Learning Rate. ')
if __name__ == "main":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    process = Optimizing(device, alpha=10**args.alpha, beta=10**args.beta, iterations=args.iteration, lr=args.learnrate)
    result_image = process.style_transfer()
    process.save_result()
