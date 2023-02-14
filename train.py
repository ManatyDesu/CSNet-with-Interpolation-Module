import os
import copy
import torch
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import make_kernels
from utils import make_inputs_imgs
from utils import open_img
from utils import calc_psnr
from models import IPM
import torch.optim as optim

# input(inputs_imgs) -> [CNN] -> video frames -> [initial reconst] -> initial image -> [deep reconst] -> output()
torch.manual_seed(0)
# -------------- 前準備 --------------
# ------ input(inputs_imgs) の作成 ------
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sr = 0.5
M = 545
blk_size, num_inputs, N, H, W = 33, 20000, 1, 264, 264
device = 'cuda' if torch.cuda.is_available else 'cpu'
#'''
bernoulli_weights = scipy.io.loadmat('')
bernoulli_weights = bernoulli_weights['sampling_matrix']
bernoulli_weights = torch.from_numpy(bernoulli_weights).float().to(device)
weights = bernoulli_weights[:M,:]
kernels = torch.zeros(M, 1, blk_size, blk_size).to(device)
for i in range(M):
    kernel = weights[i,:]
    kernel = kernel.reshape(blk_size, blk_size)
    kernels[i,:,:,:] = kernel

sr = 0.5                #here
model_dir = ""          #here
checkpoint_dir = ""     #here
load_model_path = None

max_epoch, batch_size = 4000, 128


model = IPM().to(device)
if load_model_path:
    model.load(load_model_path)
inputs_imgs = torch.zeros((num_inputs, N, H, W)).to(device) #kernelsとCNNするための入力

video_frames = torch.zeros((num_inputs, M, int(H/33), int(W/33))).to(device) #Initial Reconstructionへの入力

# ------ input(inputs_imgs) -> [CNN] -> video frames ------
#カーネル, inputs_imgsの作成
inputs_imgs = make_inputs_imgs(inputs_imgs)
#video_framesの作成
print(inputs_imgs.shape)
print(kernel.shape)
target = F.conv2d(inputs_imgs, kernels, bias=None, stride=33, padding=0)

# ------ DataLoaderの作成 ------
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = torch.utils.data.TensorDataset(inputs_imgs, target)
n_train = int(num_inputs*0.9)
n_val = num_inputs - n_train
X_train, X_val = torch.utils.data.random_split(dataset, [n_train, n_val])

train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size)

# ------ video frames -> [initial reconst] -> initial image -> [deep reconst] -> output ------
psnr_list = []
best_psnr = 0.0
for epoch in range(max_epoch):
    print(f"epoch: {epoch}")
    model.train()
    for batch in train_loader:
        x, t = batch
        x, t = x.to(device).float(), t.to(device).float()
        optimizer.zero_grad()
        
        y = model(x)
        y = y.to(device)
        loss = criterion(y, t)
        
        loss.backward()
        optimizer.step()
        
    model.eval()
    psnr_sum = 0.0
    count = 0
    for batch in valid_loader:
        x, t = batch
        x, t = x.to(device), t.to(device)
        
        with torch.no_grad():
            y = model(x)
        psnr_sum += calc_psnr(y, t)
        count += 1
    epoch_psnr = psnr_sum / count
    hyouka = epoch_psnr
    print(f"epoch_psnr -> {hyouka}")
    
    if epoch%2==0:
        checkpoint = copy.deepcopy(model.state_dict())
        torch.save(checkpoint, os.path.join(checkpoint_dir, "check_point12_20.pth"))
    if epoch_psnr > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr
        best_weights = copy.deepcopy(model.state_dict())
    hyouka = hyouka.to('cpu').detach().numpy().copy()
    psnr_list.append(hyouka)
print(f"best epoch -> {best_epoch}")
print(f"best psnr  -> {best_psnr}")
torch.save(best_weights, os.path.join(model_dir, 'best_model_12_20.pth'))