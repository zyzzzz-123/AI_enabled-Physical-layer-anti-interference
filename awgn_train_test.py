import time

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from models import FC_Autoencoder, channel_ob, encoder, decoder
from tools import EarlyStopping
from trainer import train, validate, test
from utils import generate_encoded_sym_dict
from channels import channel_init

def awgn_train(trainloader, valloader, device, args):
    # Define loggers
    log_writer_train = SummaryWriter('logs/train/')
    log_writer_val = SummaryWriter('logs/val/')

    # Setup the model and move it to GPU , #choice1: FC_Autoencoder   # choice2 : cnn_Autoencoder(not useful now)
    net = FC_Autoencoder(args.n_source, args.n_channel)
    # net.load_state_dict(torch.load('channelpara.ckpt'))
    net = net.to(device)
    ob_net = channel_ob(args)
    encoder_net = encoder(args)
    decoder_net = decoder(args)
    ob_net = ob_net.to(device)


    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # optimize all network parameters
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)   # Decay LR by a factor of 0.01 every 7 epochs
    criterion = nn.MSELoss()  # the target label is not one-hotted
    patience = 10  # early stopping patience; how long to wait after last time validation loss improved.
    early_stopping = EarlyStopping(patience=patience, verbose=True)  # initialize the early_stopping object
    loss_vec = []
    acc_vec = []
    best_accuracy = 0.0
    ##########
    start = time.time()
    for epoch in range(args.epochs):
        train_epoch_loss, train_epoch_acc = train(trainloader, net, optimizer, criterion, device, loss_vec, acc_vec,
                                                  args)
        val_loss, val_accuracy = validate(net, valloader, criterion, device, args)
        print('Epoch: ', epoch + 1, '| train loss: %.4f' % train_epoch_loss,
              '| train acc: %4f' % (train_epoch_acc * 100), '%', '| val loss: %.4f' % val_loss,
              '| val acc: %4f' % (val_accuracy * 100), '%')
        log_writer_train.add_scalar('Train/Loss', train_epoch_loss, epoch)
        log_writer_train.add_scalar('Train/Accuracy', train_epoch_acc, epoch)
        log_writer_val.add_scalar('Val/Loss', val_loss, epoch)
        log_writer_val.add_scalar('Val/Accuracy', val_accuracy, epoch)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print("best_accuracy: {best_accuracy}".format(best_accuracy=best_accuracy))
    time_elapsed = time.time() - start
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(net.state_dict(), 'channelpara.ckpt')  # Save trained net
    # generate_encoded_sym_dict(args.n_channel, args.k, net, device)  # Generate encoded symbols

    return net


def awgn_test(testloader, net, device, args):
    EbN0_test = torch.arange(args.EbN0dB_test_start, args.EbN0dB_test_end, args.EbN0dB_precision)  # Test parameters
    test_BLER = torch.zeros((len(EbN0_test), 1))
    for p in range(len(EbN0_test)):
        test_BLER[p] = test(net, args, testloader, device, EbN0_test[p])
        print('Eb/N0:', EbN0_test[p].numpy(), '| test BLER: %.4f' % test_BLER[p])
