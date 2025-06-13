# -*- coding: utf-8 -*-
'''
ulits_vae.py
'''

import os
import pickle

import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_moons

YEARBOOK_NUM_DOMAINS    = 40
YEARBOOK_TIME_POINTS    = [0, 3, 5, 6, 7, 13, 15, 18, 20, 21, 24, 25, 26, 27, 28,
                          31, 33, 34, 36, 38, 39, 42, 45, 48, 49, 50, 51, 53, 54,
                          55, 59, 60, 64, 65, 68, 69, 70, 73, 74, 78]
YEARBOOK_DOMAIN_LENGTHS = [154, 272, 492, 154, 307, 704, 834, 406, 387, 222, 581,
                          748, 269, 233, 272, 367, 259, 313, 487, 336, 617, 478,
                          819, 234, 637, 602, 255, 404, 1197, 497, 777, 801, 532,
                          637, 634, 656, 458, 411, 508, 306]

def rotate_twomoon(X, angle):
    """
    Rotate the two moons dataset by a given angle in radians.
    Input:
        X (torch.Tensor): Input data of shape (n_samples, 2).
        angle (float): Angle in radians to rotate the dataset.
    """
    angle = torch.tensor(angle, dtype=torch.float32)
    rotation_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                                     [torch.sin(angle), torch.cos(angle)]], dtype=torch.float32)
    X_rotated = X @ rotation_matrix
    return X_rotated

def load_twomoon(n_samples=1000, noise=0.1, angle=1/360):
    """
    Load the two moons dataset.
    Input:
        n_samples (int): Number of samples to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.
    Output:
        X (torch.Tensor): Input data of shape (n_samples, 2).
        y (torch.Tensor): Labels of shape (n_samples,).
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    X = rotate_twomoon(X, angle)
    y = torch.tensor(y, dtype=torch.long)

    return X, y



def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == 'Uniform':
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise


def dataset_preparation(args, num_instance=220):
    if args.dataset == 'Portrait':
        pkl_path = f'./data/{getattr(args, "portrait_pkl", "portraits_original.pkl")}'

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Portrait dataset requires '{pkl_path}' file, but it was not found.")

        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f)

        x_all = obj['data']
        y_all = obj['label']
        
        # print(f"Loaded Portrait dataset: {len(x_all)} domains")
        # for i, (x, y) in enumerate(zip(x_all, y_all)):
        #     print(f"  Domain {i}: {len(x)} samples")
        # exit(0)
        
        dataloaders = []
        for X_np, Y_np in zip(x_all, y_all):
            # Flatten image tensors from (N, 1, 32, 32) to (N, 1024)
            X_tensor = torch.tensor(X_np, dtype=torch.float32).view(X_np.shape[0], -1)
            Y_tensor = torch.tensor(Y_np, dtype=torch.long)
            dataset = TensorDataset(X_tensor, Y_tensor)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            dataloaders.append(loader)

        return dataloaders

    if args.dataset.lower() == 'yearbook':
        # File load
        yearbook_pkl = getattr(args, "yearbook_pkl", "./yearbook.pkl")
        with open(f'./data/{yearbook_pkl}', "rb") as f:
            data = pickle.load(f)
        datasets, time_points = data["datasets"], data["time_points"]

        # DataLoader
        dataloaders = []
        for i, ((X_np, Y_np), tp) in enumerate(zip(datasets, time_points)):
            # (N, 32, 32, 1) → (N, 1024) flatten
            X_np = X_np.astype(np.float32) / 255.0
            X_tensor = torch.from_numpy(X_np).view(X_np.shape[0], -1)
            Y_tensor = torch.tensor(Y_np, dtype=torch.long)
            dataset = TensorDataset(X_tensor, Y_tensor)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            dataloaders.append(loader)

        return dataloaders

    if args.dataset in ['Elec2','HousePrice','M5Hobby','M5Household','Energy']:
        # A = np.load('data/{}/processed/A.npy'.format(args.dataset))
        # U = np.load('data/{}/processed/U.npy'.format(args.dataset))
        X = np.load('data/{}/X.npy'.format(args.dataset))
        Y = np.load('data/{}/Y.npy'.format(args.dataset))

    elif args.dataset in ['Moons']:
        X = []
        Y = []
        for i in range(args.num_tasks): # 1/360 degree rotate
           X_temp, y_temp = load_twomoon(n_samples=num_instance, angle=(i/180)*(torch.pi)) # 1도씩
           X.append(X_temp)
           Y.append(y_temp)
        X = torch.stack(X, dim=0).view(-1,2) # (360*220,2)
        Y = torch.stack(Y, dim=0).view(-1,1)

    elif args.dataset in ['Shuttle']:
        data = pd.read_csv('data/shuttle/domain_data_umap_feat.csv')
        X = data.iloc[:,:-1].values.astype(np.float32) # all columns except last
        Y = data.iloc[:,-1].values.astype(np.int64)
        
    else:
        # A = np.load('data/{}/processed/A.npy'.format(args.dataset))
        # U = np.load('data/{}/processed/U.npy'.format(args.dataset))
        X = np.load('data/{}/processed/X.npy'.format(args.dataset))
        Y = np.load('data/{}/processed/Y.npy'.format(args.dataset))

        print(X.shape, Y.shape)
    
    dataloaders = []

    if args.dataset == 'Moons':
        intervals = np.arange(args.num_tasks+1)*num_instance
    elif args.dataset =='Shuttle':
        intervals = np.arange(args.num_tasks+1)*num_instance
    elif args.dataset == 'ONP':
        intervals = np.array([0,7049,13001,18725,25081,32415,39644])
    elif args.dataset == 'Elec2':
        intervals = np.array([0,670,1342,2014,2686,3357,4029,4701,5373,6045,6717,7389,8061,8733,
            9405,10077,10749,11421,12093,12765,13437,14109,14781,15453,16125,16797,17469,18141,18813,
            19485,20157,20829,21501,22173,22845,23517,24189,24861,25533,26205,26877,27549])
    elif args.dataset == 'HousePrice':
        intervals = np.array([0,2119,4982,8630,12538,17079,20937,22322])
    elif args.dataset == 'M5Hobby':
        intervals = np.array([0,323390,323390*2,323390*3,997636])
    elif args.dataset == 'M5Household':
        intervals = np.array([0,124100,124100*2,124100*3,382840])
    elif args.dataset == 'Energy':
        intervals = np.array([0,2058,2058+2160,2058+2*2160,2058+3*2160,2058+4*2160,2058+5*2160,2058+6*2160,2058+7*2160,19735])

    for i in range(len(intervals)-1):
        temp_X = X[intervals[i]:intervals[i+1]]
        temp_Y = Y[intervals[i]:intervals[i+1]]

        domain_dataset = DomainDataset(temp_X,temp_Y) # create dataset for each domain
        temp_dataloader = DataLoader(domain_dataset, batch_size=args.batch_size, 
                                     shuffle=True, num_workers=args.num_workers)
        dataloaders.append(temp_dataloader)
    
    return dataloaders

class DomainDataset(Dataset):
    """ Customized dataset for each domain"""
    def __init__(self,X,Y):
        self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X # set data
        self.Y = torch.tensor(Y, dtype=torch.float32) if not torch.is_tensor(Y) else Y # set lables

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]    # return list of batch data [data, labels]