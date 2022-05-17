import torch.nn as nn
import channels
import torch
from channels import plot_fft_one_channel_2, channel_choose, encoded_normalize
from utils import calculateSNR

class ob_autoencoder(nn.Module):
    def __init__(self, args):
        self.args = args

        super(ob_autoencoder, self).__init__()
        self.ob_encoder = nn.Sequential(
            nn.Linear(in_features=(self.args.observe_length*2), out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2, bias=True),

        )
        self.ob_decoder = nn.Sequential(
            nn.Linear(in_features=(self.args.observe_length * 2), out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2, bias=True),
        )
        self.encoder = nn.Sequential(
            nn.Linear(in_features=(self.args.n_source + self.args.number_interfs), out_features=8, bias=True),
            # nn.PReLU(),
            # nn.Linear(in_features=32, out_features=8, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=2, bias=True),
            )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=(self.args.n_channel*2 + self.args.number_interfs), out_features=32, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.args.n_source, bias=True),
            # nn.Softmax(dim=1),
        )


    def forward(self, x, interf_500, interf_500_2, interf_64, step_total):
        batch_size, _ = x.shape

        ## observation net
        interf_500 = interf_500.reshape(batch_size,self.args.observe_length*2)
        interf_500_2 = interf_500_2.reshape(batch_size,self.args.observe_length*2)
        ob_result = self.ob_encoder(interf_500)
        ob_result2  = self.ob_decoder(interf_500_2)

        ## encoder
        encode_input = torch.cat([ob_result,x],dim=1)
        encoded = self.encoder(encode_input).unsqueeze(1)

        ## channel
        encoded_normalized = encoded_normalize(encoded, self.args)  # make encoded's max amp equal to 1.
        encoded_choose = channel_choose(encoded_normalized,self.args)  # encoded_choose shape: (bs, 64, 2).
                                                                        # And only [:, 8, :] is not 0 if one channel
        transmitted_complex = torch.complex(encoded_choose[:, :, 0], encoded_choose[:, :, 1])
        transmitted_time = torch.fft.ifft(transmitted_complex,dim=1)
        transmitted_time_ = torch.stack([torch.real(transmitted_time), torch.imag(transmitted_time)], dim=2)
        transmitted = transmitted_time_ + interf_64.detach()  # add interference to signal

        ## decoder
        decoder_input =  torch.cat([transmitted[:,:,0],transmitted[:,:,1],ob_result2],dim=1)
        output = self.decoder(decoder_input)

        # plot transmitted and interf in frequency domain
        if ((step_total > 0) and (step_total % self.args.print_step == 0)):
            calculateSNR(transmitted_time_, interf_64)
            plot_fft_one_channel_2(transmitted_time_, interf_64)
            print(output[0])

        return output