# -*- coding: utf-8 -*-
'''
model.py
'''
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, data_size, device, args):
        super(RNN, self).__init__()
        self.init_lin_h = nn.Linear(args.noise_dim, args.rnn_latent_dim)
        self.init_lin_c = nn.Linear(args.noise_dim, args.rnn_latent_dim)
        self.init_input = nn.Linear(args.noise_dim, args.rnn_latent_dim)

        self.rnn = nn.LSTM(args.rnn_latent_dim, args.rnn_latent_dim, args.num_rnn_layer)

        input_dim = data_size
        vae_hidden_dim = args.vae_hidden_dim
        vae_latent_dim = args.vae_latent_dim

        self.vae_hidden_dim = args.vae_hidden_dim
        self.vae_latent_dim = args.vae_latent_dim


        encoder_params = input_dim * vae_hidden_dim + vae_hidden_dim + \
                         vae_hidden_dim * vae_latent_dim + vae_latent_dim + \
                         vae_hidden_dim * vae_latent_dim + vae_latent_dim

        decoder_params = vae_latent_dim * vae_hidden_dim + vae_hidden_dim + \
                         vae_hidden_dim * input_dim + input_dim

        vae_param_count = encoder_params + decoder_params
        self.vae_param_count = vae_param_count  

        self.lin_transform_down = nn.Sequential(
            nn.Linear(args.rnn_latent_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, vae_param_count)
        )

        self.lin_transform_up = nn.Sequential(
            nn.Linear(vae_param_count, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_latent_dim)
        )

        self.num_rnn_layer = args.num_rnn_layer
        self.data_size = data_size
        self.device = device

    def vae_parameter_split(self, E):
        input_dim = self.data_size
        vae_hidden_dim = self.vae_hidden_dim
        vae_latent_dim = self.vae_latent_dim
        # Split the parameters from E
        i = 0
        def take(n):
            nonlocal i
            r = E[:, i:i+n]
            i += n
            return r

        # Encoder
        W1_enc = take(input_dim * vae_hidden_dim).view(input_dim, vae_hidden_dim)
        b1_enc = take(vae_hidden_dim)
        W_mu = take(vae_hidden_dim * vae_latent_dim).view(vae_hidden_dim, vae_latent_dim)
        b_mu = take(vae_latent_dim)
        W_logvar = take(vae_hidden_dim * vae_latent_dim).view(vae_hidden_dim, vae_latent_dim)
        b_logvar = take(vae_latent_dim)

        # Decoder
        W1_dec = take(vae_latent_dim * vae_hidden_dim).view(vae_latent_dim, vae_hidden_dim)
        b1_dec = take(vae_hidden_dim)
        W2_dec = take(vae_hidden_dim * input_dim).view(vae_hidden_dim, input_dim)
        b2_dec = take(input_dim)
        return {
                    'W1_enc': W1_enc,
                    'b1_enc': b1_enc,
                    'W_mu': W_mu,
                    'b_mu': b_mu,
                    'W_logvar': W_logvar,
                    'b_logvar': b_logvar,
                    'W1_dec': W1_dec,
                    'b1_dec': b1_dec,
                    'W2_dec': W2_dec,
                    'b2_dec': b2_dec
                }

    
    def vae_forward(self, X, params):
        # Encoding
        h = torch.relu(torch.mm(X, params['W1_enc']) + params['b1_enc'])
        mu = torch.mm(h, params['W_mu']) + params['b_mu']
        logvar = torch.mm(h, params['W_logvar']) + params['b_logvar']
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decoding
        h_dec = torch.relu(torch.mm(z, params['W1_dec']) + params['b1_dec'])
        x_hat = torch.mm(h_dec, params['W2_dec']) + params['b2_dec']

        return x_hat, mu, logvar

    def forward(self, X, z, E=None, hidden=None):
        if hidden == None and E == None:
            init_c, init_h = [], []
            for _ in range(self.num_rnn_layer):
                init_c.append(torch.tanh(self.init_lin_h(z)))
                init_h.append(torch.tanh(self.init_lin_c(z)))
            # Initialize hidden inputs for the LSTM
            hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
        
            # Initialize an input for the LSTM
            inputs = torch.tanh(self.init_input(z))
        else:
            inputs = self.lin_transform_up(E)

        out, hidden = self.rnn(inputs.unsqueeze(0), hidden)

        E = self.lin_transform_down(out.squeeze(0))
        '''
        m_1, m_2, bias = self.nn_construction(E)
        
        pred = torch.relu(torch.mm(X, m_1))
        pred = torch.sigmoid(torch.add(torch.mm(pred, m_2), bias))
        '''
        
        params = self.vae_parameter_split(E)
        x_hat, mu, logvar = self.vae_forward(X, params)
        
        return E, hidden, x_hat, mu, logvar
    
class Predictor(nn.Module):
    def __init__(self, data_size, args):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(data_size, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.out = nn.Linear(args.hidden_dim, 1)
        self.data_size = data_size
    def forward(self, X):
        x = torch.relu(self.fc1(X))
        x = torch.relu(self.fc2(x))
        logits = self.out(x)
        probs = torch.sigmoid(logits)
        return probs, logits