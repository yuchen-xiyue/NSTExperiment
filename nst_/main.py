# for command line
from training import *
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-i', '--iteration', type=int, help='Max iteration for optimization')
parser.add_argument('-a', '--alpha', type=int, default=-2, help='The exponential index of content loss weight')
parser.add_argument('-b', '--beta',  type=int, default=9,  help='The exponential index of style loss weight')
parser.add_argument('-l', '--learnrate', type=float, help='Learning Rate. ')

global STYLE_WEIGHT, CONTENT_WEIGHT
args = parser.parse_args()
if args.iteration:
    global MAX_ITERATIONS
    MAX_ITERATIONS = args.iteration
alpha, beta = 10**args.alpha, 10**args.beta
CONTENT_WEIGHT, STYLE_WEIGHT = alpha * CONTENT_WEIGHT, beta * STYLE_WEIGHT
if args.learnrate:
    global LEARNING_RATE
    LEARNING_RATE = args.learnrate
result_image = style_transfer()
show_img(result_image).save(RESULT_PATH)
