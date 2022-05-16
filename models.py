import torch.nn as nn
import channels
import torch


class FC_Autoencoder(nn.Module):
    def __init__(self, k, n_channel):
        self.k = k
        self.n_channel = n_channel

        super(FC_Autoencoder, self).__init__()
        self.transmitter = nn.Sequential(
            nn.Linear(in_features=self.k, out_features=self.n_channel, bias=True),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(in_features=self.n_channel, out_features=self.n_channel, bias=True),
            # nn.ReLU(inplace=True),
            # nn.PReLU(),
            # nn.Linear(in_features=self.n_channel, out_features=self.n_channel, bias=True))
            )
        self.receiver = nn.Sequential(
            nn.Linear(in_features=self.n_channel, out_features= self.n_channel , bias=True),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(in_features=self.n_channel, out_features=self.k, bias=True),
            # nn.ReLU(inplace=True),
            # nn.PReLU(),
            # nn.Linear(in_features=self.k, out_features=self.k, bias=True),
        )

    def forward(self, x):
        # x_transmitted = self.transmitter(x)
        # # x_normalized = self.energy_normalize(x_transmitted)
        # x_noisy = channels.awgn(x_transmitted,args)  # Gaussian Noise
        # x = self.receiver(x_noisy)
        # x = x.to(device)
        return x




class channel_ob(nn.Module):
    """
    channel observation module
    """
    def __init__(self, args):
        self.args = args

        super(channel_ob, self).__init__()
        self.complex = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=49, padding=24)
        self.observation = nn.Sequential(
            nn.Linear(in_features=1000, out_features=8, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.args.number_interfs, bias=True),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        batch_size, _ ,_ = x.shape
        x = x.reshape(batch_size, self.args.observe_length*2)
        x = self.observation(x)
        return x


class encoder(nn.Module):
    """
    encoder module
    input: [bz, N_OFDM_SYMS, N_OFDM_SYMS+number_interfs]
    output: [bz, n_channel, 2] (dtype = float)
    """
    def __init__(self, args):
        self.args = args

        super(encoder, self).__init__()
        # self.encoder1 = nn.Sequential(
        #     nn.Conv1d(in_channels=1,out_channels=2,kernel_size=49,stride=1,padding = 24),
        #     nn.Conv1d(in_channels=2, out_channels=1, kernel_size= 49, stride=1 , padding= 24),
        # )
        self.encoder2 = nn.Sequential(
            nn.Linear(in_features=(self.args.n_source+ self.args.number_interfs), out_features=8, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=2, bias=True),
            )


    def forward(self, x):
        batch_size, _ = x.shape
        # x = x.unsqueeze(1)
        # x = self.encoder1(x)
        # x = x.squeeze(1)
        x  = self.encoder2(x).unsqueeze(1)
        # out = torch.repeat_interleave(x, self.args.n_channel, 1)
        return x

class decoder(nn.Module):
    """
    decoder module
    input : (n_channel + number_interfs) * 2
    """
    def __init__(self, args):
        self.args = args

        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=(self.args.n_channel*2 + self.args.number_interfs), out_features=8, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=8, out_features=self.args.n_source, bias=True),
            nn.Softmax(dim=1),
        )
        # self.decomplex =  nn.Sequential(
        #     nn.Linear(in_features=2, out_features=1, bias=True),
        #     nn.PReLU(),
        #

    def forward(self, x):
        batch_size, _ = x.shape
        out = self.decoder(x)
        # out = torch.argmax(x,dim=1)
        return out