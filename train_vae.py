# -*- coding: utf-8 -*-
'''
train_vae
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse
import pickle

# Import model
from model import RNN
# Import functions
from utils_vae import dataset_preparation, make_noise

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./generated', exist_ok=True)


log_file = 'logs/log_{}.log'.format(datetime.datetime.now().strftime("%Y_%B_%d_%I-%M-%S%p"))
open(log_file, 'a').close()

# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log(str): logger.info(str)


log('Is GPU available? {}'.format(torch.cuda.is_available()))
#print('Is GPU available? {}'.format(torch.cuda.is_available()))

parser = argparse.ArgumentParser(description="DomainGen_Graph")

datasets = ['ONP', 'Moons', 'MNIST', 'Elec2', 'Portrait', 'Yearbook']
parser.add_argument("--dataset", default="Moons", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
parser.add_argument("--portrait_pkl", default="portraits_original.pkl", type=str,
                    help="Path to the portrait dataset .pkl file.")
parser.add_argument("--yearbook_pkl", default="yearbook.pkl", type=str,
                    help="Path to the yearbook dataset .pkl file.")# Hyper-parameters
parser.add_argument("--noise_dim", default=16, type=int,
                    help="the dimension of the LSTM input noise.")
parser.add_argument("--num_rnn_layer", default=10, type=int,
                    help="the number of RNN hierarchical layers.")

parser.add_argument("--rnn_latent_dim", default=16, type=int,
                    help="the latent dimension of RNN variables.")

parser.add_argument("--vae_latent_dim", default=4, type=int,
                    help="the latent dimension of RNN variables.")

parser.add_argument("--rnn_hidden_dim", default=50, type=int,
                    help="the latent dimension of RNN variables.")

parser.add_argument("--vae_hidden_dim", default=16, type=int,
                    help="the latent dimension of RNN variables.")

parser.add_argument("--Beta_vae", default=0.001, type=float,
                    help="the latent dimension of RNN variables.")

parser.add_argument("--noise_type", choices=["Gaussian", "Uniform"], default="Gaussian",
                    help="The noise type to feed into the generator.")

parser.add_argument("--num_workers", default=0, type=int,
                    help="the number of threads for loading data.")
parser.add_argument("--epoches", default=50, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--batch_size", default=64, type=int,
                    help="the number of epoches for each task.")
parser.add_argument("--learning_rate", default=1e-4, type=float,
                    help="the unified learning rate for each single task.")

parser.add_argument("--is_test", default=True, type=bool,
                    help="if this is a testing period.")

parser.add_argument("--gpu", default=3, type=int,
                    help="GPU device id to use (default: 0)")

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

def train(dataloader, optimizer, rnn_unit, args, task_id=0, input_E=None, input_hidden=None):
    E = input_E
    hidden = input_hidden
    log("Start Training on Domain {}...".format(task_id))
    all_X, all_x_hat, all_Y = [], [], []

    for epoch in range(args.epoches):
        recon_losses = []
        with tqdm(dataloader, unit="batch") as tepoch:
            for X, Y in tepoch:
                tepoch.set_description("Task_ID: {} Epoch {}".format(task_id, epoch))
                
                X = X.float().to(device)
                initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
                
                #  Training on Single Domain
                rnn_unit.train()
                optimizer.zero_grad()
                E, hidden, x_hat, mu, logvar = rnn_unit(X, initial_noise, E, hidden)
                E = E.detach()
                
                hidden = tuple([i.detach() for i in hidden])

                recon_loss = F.mse_loss(x_hat, X, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + args.Beta_vae * kl_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
                
                recon_losses.append(recon_loss.item())
                tepoch.set_postfix(loss=loss.item(), recon_loss=recon_loss.item())
                # epoch==args.epoches-1 조건을 넣어도 됨
                if epoch == 0:
                    all_X.append(X.cpu())
                    all_x_hat.append(x_hat.detach().cpu())
                    all_Y.append(Y.cpu())
                    
    # Save recon result per domain
    X_cat = torch.cat(all_X, dim=0).numpy()
    xhat_cat = torch.cat(all_x_hat, dim=0).numpy()
    Y_cat = torch.cat(all_Y, dim=0).numpy()

    np.save(f"./generated/{args.dataset}_x_task_{task_id}.npy", X_cat)
    np.save(f"./generated/{args.dataset}_xhat_task_{task_id}.npy", xhat_cat)
    np.save(f"./generated/{args.dataset}_y_task_{task_id}.npy", Y_cat)

    return E, hidden, rnn_unit
    

def evaluation(dataloader, rnn_unit, args, input_E, input_hidden):
    rnn_unit.eval()
    E = input_E
    hidden = input_hidden
    test_recon_loss = []
    log("Start Testing...")
    all_X, all_x_hat, all_Y = [], [], []

    with tqdm(dataloader, unit="batch") as tepoch:
        for X, Y in tepoch:                
            X = X.float().to(device)
            initial_noise = make_noise((1, args.noise_dim), args.noise_type).to(device)
            with torch.no_grad():
                E, hidden, x_hat, mu, logvar = rnn_unit(X, initial_noise, E, hidden)
                recon_loss = F.mse_loss(x_hat, X, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss

                test_recon_loss.append(recon_loss.item())
                tepoch.set_postfix(loss=loss.item(), recon_loss=recon_loss.item())
                all_x_hat.append(x_hat.cpu())
                all_X.append(X.cpu())
                all_Y.append(Y.cpu())

    # 저장
    X_all = torch.cat(all_X, dim=0).numpy()
    x_hat_all = torch.cat(all_x_hat, dim=0).numpy()
    Y_all     = torch.cat(all_Y,    dim=0).numpy()

    np.save(f"./generated/{args.dataset}_x_eval.npy", X_all)
    np.save(f"./generated/{args.dataset}_xhat_eval.npy", x_hat_all)
    np.save(f"./generated/{args.dataset}_y_eval.npy",    Y_all)

    log("Average Testing Reconstruction loss is {}".format(np.mean(test_recon_loss)))


def main(arsgs):
    os.makedirs('generated', exist_ok=True)
    
    log('use {} data'.format(args.dataset))
    log('-'*40)

    if args.dataset == 'Portrait':
        with open(f'./data/{args.portrait_pkl}', 'rb') as f:
            obj = pickle.load(f)
            num_tasks = len(obj['data']) 
            data_size = obj['data'][0].reshape(obj['data'][0].shape[0], -1).shape[1]  # Flatten
        num_instances = None
    elif args.dataset == 'Yearbook':
        with open(f'./data/{args.yearbook_pkl}', 'rb') as f:
            obj = pickle.load(f)
        datasets, time_points = obj['datasets'], obj['time_points']

        num_tasks = len(datasets)  # should be 40

        sample_shape = datasets[0][0].shape  # e.g. (154,32,32,1)
        data_size = int(np.prod(sample_shape[1:]))  # 32*32*1 = 1024
        num_instances = None
        print(num_tasks, sample_shape, data_size)
    elif args.dataset == 'Moons':
        num_tasks=180 # source + intermediate + target
        data_size=2
        num_instances=500 # number of instances per task
    elif args.dataset == 'Shuttle':
        num_tasks=28 # source + intermediate + target
        data_size=2
        num_instances=2000 # number of instances per task    
    elif args.dataset == 'MNIST':
        num_tasks=11
        data_size=2
        num_instances=200
    elif args.dataset == 'ONP':
        num_tasks=6
        data_size=58
        num_instances=None
    elif args.dataset == 'Elec2':
        num_tasks=41
        data_size=8
        num_instances=None
    # Defining dataloaders for each domain
    args.num_tasks = num_tasks
    
    dataloaders = dataset_preparation(args, num_instances)
    rnn_unit = RNN(data_size, device, args).to(device)
    
    # Loss and optimizer
    optimizer = torch.optim.Adam(rnn_unit.parameters(), lr=args.learning_rate)

    starting_time = time.time()
    
    # Training
    Es, hiddens = [None], [None]
    
    
    for task_id, dataloader in enumerate(dataloaders[:-1]):
        E, hidden, rnn_unit = train(dataloader, optimizer, rnn_unit, args, task_id, Es[-1], hiddens[-1])
        Es.append(E)
        hiddens.append(hidden)
        print("========== Finished Task #{} ==========".format(task_id))
        

    ending_time = time.time()

    print("Training time:", ending_time - starting_time)
    
    # Testing
    evaluation(dataloaders[-1], rnn_unit, args, Es[-1], hiddens[-1])
        

if __name__ == "__main__":
    print("Start Training...")
    
    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    main(args)