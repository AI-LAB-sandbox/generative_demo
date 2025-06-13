# -*- coding: utf-8 -*-
'''
train_classifier
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import pickle
import random
import os
import logging
import time
import datetime
from tqdm import tqdm
import argparse
import copy
import pdb
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader


# Import model
from model import Predictor
# Import functions
from utils_vae import dataset_preparation, make_noise

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./checkpoint', exist_ok=True)

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DomainGen_Graph")

datasets = ['ONP', 'Moons', 'MNIST', 'Elec2', 'Portrait', 'Yearbook']
parser.add_argument("--dataset", default="Moons", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
parser.add_argument("--portrait_pkl", default="portraits_original.pkl", type=str,
                    help="Path to the portrait dataset .pkl file.")
parser.add_argument("--yearbook_pkl", default="yearbook.pkl", type=str,
                    help="Path to the yearbook dataset .pkl file.")

# Hyper-parameters
parser.add_argument("--hidden_dim", default=64, type=float,
                    help="the hidden dimension of predictor.")

parser.add_argument("--num_workers", default=0, type=int,
                    help="the number of threads for loading data.")

parser.add_argument("--epoches", default=100, type=int,
                    help="the number of epoches for each task.")

parser.add_argument("--st_epoches", default=50, type=int,
                    help="the number of epoches for each task.")

parser.add_argument("--batch_size", default=16, type=int,
                    help="the number of epoches for each task.")

parser.add_argument("--learning_rate", default=1e-4*5, type=float,
                    help="the unified learning rate for each single task.")

parser.add_argument("--is_test", default=True, type=bool,
                    help="if this is a testing period.")

parser.add_argument("--gpu", default=3, type=int,
                    help="GPU device id to use (default: 0)")

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    print(f"[Info] Setting seed to {seed} for all relevant libraries.")

    # Python 내장 random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch (CPU, CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 대응

    # PyTorch 확실한 재현성 옵션
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # 성능은 약간 희생

    # 환경 변수 (hash seed 등)
    os.environ["PYTHONHASHSEED"] = str(seed)



def load_generated_target_dataset(args):
    x_hat = np.load(f"generated/{args.dataset}_xhat_eval.npy")
    X_tensor = torch.tensor(x_hat, dtype=torch.float32)
    dummy_labels = torch.zeros(X_tensor.size(0)) 
    dataset = TensorDataset(X_tensor, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def evaluate_classifier_on_target(classifier, dataloader):
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.float().to(device)
            if args.dataset == "Moons":
                Y = Y.float().to(device)
            else:
                Y = Y.float().to(device).unsqueeze(1)
            probs, _ = classifier(X)
            preds = (probs > 0.5).float()
            correct += (preds == Y).sum().item()
            total += Y.size(0)
    acc = correct / total
    log(f"🎯 Evaluation Accuracy: {acc:.4f}")
    return acc

def main(arsgs):

    log('use {} data'.format(args.dataset))
    log('-'*40)
    if args.dataset == 'Portrait':
        with open(f'./data/{args.portrait_pkl}', 'rb') as f:
            obj = pickle.load(f)
            num_tasks = len(obj['data'])  # 도메인 개수
            data_size = obj['data'][0].reshape(obj['data'][0].shape[0], -1).shape[1]  # Flatten
        num_instances = None
    elif args.dataset == 'Yearbook':
        with open(f'./data/{args.yearbook_pkl}', 'rb') as f:
            obj = pickle.load(f)
        datasets, time_points = obj['datasets'], obj['time_points']

        # 2) 도메인·크기 설정
        num_tasks = len(datasets)  # should be 40
        # 각 X_np: (N,32,32,1) 이므로 flatten 차원 계산
        sample_shape = datasets[0][0].shape  # e.g. (154,32,32,1)
        data_size = int(np.prod(sample_shape[1:]))  # 32*32*1 = 1024
        num_instances = None
        print(num_tasks, sample_shape, data_size)
    elif args.dataset == 'Moons':
        args.num_tasks=180
        data_size=2
        num_instances=500
    elif args.dataset == 'Shuttle':
        args.num_tasks=28
        data_size=2
        num_instances=2000
    elif args.dataset == 'MNIST':
        args.num_tasks=11
        data_size=2
        num_instances=200
    elif args.dataset == 'ONP':
        args.num_tasks=6
        data_size=58
        num_instances=None
    elif args.dataset == 'Elec2':
        args.num_tasks=41
        data_size=8
        num_instances=None
    
    # Defining dataloaders for each domain
    dataloaders = dataset_preparation(args, num_instances)
 
    starting_time = time.time()

    prev_classifier = None

    for task_id, dataloader in enumerate(dataloaders[:-1]):
        classifier = Predictor(data_size, args).to(device)
        optimizer_inner = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        classifier.train()
        total_loss = 0.0
        if task_id == 0:
            # Supervised learning
            for epoch in range(args.epoches):
                for X, Y in dataloader:
                    X = X.float().to(device)
                    if args.dataset == 'Moons':
                        Y = Y.float().to(device)
                    else:
                        Y = Y.float().to(device).unsqueeze(1)

                    optimizer_inner.zero_grad()
                    probs, logits = classifier(X)
                    loss = F.binary_cross_entropy(probs, Y)
                    loss.backward()
                    optimizer_inner.step()
                    total_loss += loss.item()
                    # print(Y)
            prev_classifier = copy.deepcopy(classifier)  # self-training에 쓸 모델 저장
            initial_classifier = copy.deepcopy(classifier)  # 초기 모델 저장


            log(f"Task {task_id} epoch {epoch}: Supervised learning completed with loss {loss:.4f}")
            log(f"Task {task_id} epoch {epoch}: Supervised learning completed with toal loss {total_loss:.4f}")

        else:        
            # Self-training
            for epoch in range(args.st_epoches):
                for X, Y in dataloader:
                    X = X.float().to(device)
                    if args.dataset == 'Moons':
                        Y = Y.float().to(device)
                    else:
                        Y = Y.float().to(device).unsqueeze(1)


                    with torch.no_grad():
                        probs, _ = prev_classifier(X)
                        pseudo_labels = (probs > 0.5).float()
                        # print(pseudo_labels)

                    optimizer_inner.zero_grad()
                    probs, logits = classifier(X)
                    # confidence = torch.clamp((probs - 0.5).abs() * 2, min=0.5) # or probs, depending on strategy
                    loss = F.binary_cross_entropy(probs, pseudo_labels, reduction='none')
                    weighted_loss = (loss).mean()

                    weighted_loss.backward()
                    optimizer_inner.step()
                    total_loss += weighted_loss.item()

                    correct = 0
                    total = 0
                    if epoch == args.st_epoches - 1:
                        correct = (pseudo_labels == Y).float().sum().item()
                        total = Y.size(0)
                        pseudo_acc = correct / total
                        # log(f"🔍 Pseudo-label Accuracy on Domain {task_id}: {pseudo_acc:.4f}")
                        # 마지막 epoch에 정확도 계산
                        preds = (probs > 0.5).float()
                        correct += (preds == Y).sum().item()
                        total += Y.size(0)

                
            log(f"Task {task_id} epoch {epoch}: Self-training completed with loss {weighted_loss:.4f}")
            log(f"Task {task_id} epoch {epoch}: Self-training completed with loss {total_loss:.4f}")
            log(f"Task {task_id} epoch {epoch}: Average Training Accuracy: {correct / total:.4f}")

            print(f"========== Finished Task #{task_id} ==========")
            # Save the model after each task
            prev_classifier = copy.deepcopy(classifier)
            prev_classifier.eval()

        if task_id % 5 == 0:
            save_path = os.path.join('./checkpoint', f"{args.dataset}_classifier_task_{task_id}.pt")
            torch.save(classifier.state_dict(), save_path)

        xhat_task = np.load(f"generated/{args.dataset}_xhat_task_{task_id}.npy")
        X_task_tensor = torch.tensor(xhat_task, dtype=torch.float32).to(device)
        prev_classifier.eval()
        with torch.no_grad():
            probs_task, _ = prev_classifier(X_task_tensor)
            yhat_task = (probs_task > 0.5).float().cpu().numpy()
        np.save(f"generated/{args.dataset}_yhat_task_{task_id}.npy", yhat_task)
        
    target_dataloader = load_generated_target_dataset(args)


    last_classifier = Predictor(data_size, args).to(device)
    optimizer_inner = torch.optim.Adam(last_classifier.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    for epoch in range(args.st_epoches):
        for X, _ in target_dataloader:
            X = X.float().to(device)
            
            with torch.no_grad():
                probs, _ = prev_classifier(X)
                pseudo_labels = (probs > 0.5).float()

            optimizer_inner.zero_grad()
            probs, logits = last_classifier(X)
            # confidence = torch.clamp((probs - 0.5).abs() * 2, min=0.5) # or probs, depending on strategy
            loss = F.binary_cross_entropy(probs, pseudo_labels, reduction='none')
            weighted_loss = loss.mean()

            weighted_loss.backward()
            optimizer_inner.step()
    
    log("✔ Target domain fine-tuned using generated data.")

    ending_time = time.time()

    print("Training time:", ending_time - starting_time)
    
    # Testing
    last_classifier.eval()
    x_hat_eval    = np.load(f"generated/{args.dataset}_xhat_eval.npy")
    X_eval_tensor = torch.tensor(x_hat_eval, dtype=torch.float32).to(device)

    last_classifier.eval()

    with torch.no_grad():
        probs_last, _ = last_classifier(X_eval_tensor)
        yhat_eval     = (probs_last > 0.5).float().cpu().numpy()

    np.save(f"generated/{args.dataset}_yhat_eval.npy", yhat_eval)

    save_path = os.path.join('./checkpoint', f"{args.dataset}_last_classifier.pt")
    torch.save(last_classifier.state_dict(), save_path)


    evaluate_classifier_on_target(last_classifier, dataloaders[-1])
    log(f"Total test samples: {total}")

    print("Evaluation for initial classifier on target domain")
    evaluate_classifier_on_target(initial_classifier, dataloaders[-1])


if __name__ == "__main__":
    print("Start Training...")

    set_seed(42)  # Set a fixed seed for reproducibility
    
    # Initialize the time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    main(args)