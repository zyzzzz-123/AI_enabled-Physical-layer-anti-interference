"""
Yuzhe Zhang;
AI_enabled physical layer anti-interference project
github : https://github.com/zyzzzz-123/AI_enabled-Physical-layer-anti-interference
"""
import torch

from awgn_train_test import awgn_train, awgn_test
from utils import prepare_data
import argparse

# User parameters
parser = argparse.ArgumentParser()
parser.add_argument('-n_channel', type=float, default=64)  # length of channel
parser.add_argument('-n_source', type=float, default=2)  # length of initial data
parser.add_argument('-N_OFDM_SYMS', type=int, default=50)  # numbers of OFDM_systems
parser.add_argument('-number_interfs', type=int, default=2)  # numbers of interference catagories
parser.add_argument('-observe_length', type=int, default=500)  # length of channel observation
parser.add_argument('-channel_length', type=int, default=64)  # length of channel length
parser.add_argument('-AMP', type=float, default= 0.001)                # relative amplitude of interference default = 0.25
parser.add_argument('-AMP_n', type=float, default=0.0001)                # relative amplitude of AWGN
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
# parser.add_argument('-modulation', choices = ['BPSK','QPSK','8PSK','16QAM','64QAM','256QAM'], default='BPSK')
# parser.add_argument('-coding', choices=['SingleParity_4_3','Hamming_7_4','Hamming_15_11','Polar_16_4', 'EGolay_24_12'],default='Hamming_7_4')
parser.add_argument('-train_set_size', type=int, default=22500)
parser.add_argument('-val_set_size', type=int, default=3000)


### channel switches and
parser.add_argument('-AWGN', type=bool, default=True)
parser.add_argument('-CFO', type=bool, default=False)
parser.add_argument('-SFO', type=bool, default=False)
### channel adjustment details
parser.add_argument('-SFO_delta_max', type=float , default=0)   # SFO，采样时间偏移比例，对调制信号
parser.add_argument('-SFO_delta_max_500', type=float , default=0)   # SFO，采样时间偏移比例，对500点干扰
parser.add_argument('-SFO_delta_max_64', type=float , default=0) # SFO，采样时间偏移比例，对64点干扰
parser.add_argument('-CFO_delta_max', type=float , default=0)   # CFO，采样时间偏移比例，对调制信号
parser.add_argument('-CFO_delta_max_500', type=float , default=0)   # CFO，采样时间偏移比例，对500点干扰
parser.add_argument('-CFO_delta_max_64', type=float , default=0) # CFO，采样时间偏移比例，对64点干扰


####  train parameters  #####
parser.add_argument('-learning_rate', type=float, default=1e-3)
parser.add_argument('-batch_size', type=int, default=50)
parser.add_argument('-dropout', type=float, default=0.0)
parser.add_argument('-EbN0_dB_train', type=float, default=5.0)
parser.add_argument('-EbN0dB_test_start', type=float, default=30.0)
parser.add_argument('-EbN0dB_test_end', type=float, default=11.5)
parser.add_argument('-EbN0dB_precision', type=int, default=0.5)
parser.add_argument('-epochs', type=int, default=10000)
parser.add_argument('-print_step', type=int, default=200)
args = parser.parse_args()
# R = args.n_source / args.n_channel


# CUDA for PyTorch - Makes sure the program runs on GPU when available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def run():
    torch.backends.cudnn.benchmark = True  # Make sure torch is accessing CuDNN libraries

    # Prepare data
    traindataset, trainloader, train_labels = prepare_data(args, "train")  # Train data
    valdataset, valloader, val_labels = prepare_data(args, "val")  # Validation data

    # Training
    trained_net = awgn_train(trainloader, valloader, device, args)

    # Testing
    # awgn_test(testloader, trained_net, device, args)


if __name__ == '__main__':
    run()
