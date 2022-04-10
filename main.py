import torch

from awgn_train_test import awgn_train, awgn_test
from utils import prepare_data
import argparse

# User parameters
parser = argparse.ArgumentParser()
parser.add_argument('-n_channel', type=float, default=32)  # length of channel
parser.add_argument('-n_source', type=float, default=8)  # length of initial data
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
# parser.add_argument('-modulation', choices = ['BPSK','QPSK','8PSK','16QAM','64QAM','256QAM'], default='BPSK')
# parser.add_argument('-coding', choices=['SingleParity_4_3','Hamming_7_4','Hamming_15_11','Polar_16_4', 'EGolay_24_12'],default='Hamming_7_4')
parser.add_argument('-train_set_size', type=int, default=10000)
parser.add_argument('-val_set_size', type=int, default=5000)


####  train parameters  #####
parser.add_argument('-learning_rate', type=float, default=1e-3)
parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-dropout', type=float, default=0.0)
parser.add_argument('-EbN0_dB_train', type=float, default=5.0)
parser.add_argument('-EbN0dB_test_start', type=float, default=30.0)
parser.add_argument('-EbN0dB_test_end', type=float, default=11.5)
parser.add_argument('-EbN0dB_precision', type=int, default=0.5)
parser.add_argument('-epochs', type=int, default=10000)
args = parser.parse_args()
# R = args.n_source / args.n_channel


# CUDA for PyTorch - Makes sure the program runs on GPU when available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def run():
    torch.backends.cudnn.benchmark = True  # Make sure torch is accessing CuDNN libraries

    # Prepare data
    traindataset, trainloader, train_labels = prepare_data(args.train_set_size, args.n_source,
                                                           args.batch_size)  # Train data
    valdataset, valloader, val_labels = prepare_data(args.val_set_size, args.n_source,
                                                     args.batch_size)  # Validation data

    # Training
    trained_net = awgn_train(trainloader, valloader, args.val_set_size, device, args)

    # Testing
    # awgn_test(testloader, trained_net, device, args)


if __name__ == '__main__':
    run()
