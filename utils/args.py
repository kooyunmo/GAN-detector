import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, help='data directory')
    parser.add_argument('--model-path', type=str, help='model path')
    parser.add_argument('--output-path', type=str, help='output path')
    parser.add_argument('--batch-size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--gpu', type=str, default='0', help='device id')
    parser.add_argument('--visualize', action='store_true', help='ploting training loss plot')
    parser.add_argument('--result-dir', type=str, default='directory to save performance results')
    return parser.parse_args()
