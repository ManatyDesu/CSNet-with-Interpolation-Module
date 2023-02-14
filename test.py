import os
import copy
import torch
import glob
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import calc_psnr
from utils import open_img
from models import CSNet
from skimage.metrics import structural_similarity
import time
from utils import make_kernels

#first_sr -> sr (補間する)
first_sr = 0.25             #here
sr = 0.5                    #here
now = "set11"               #here
save_pth = "result/SR-050/vid4" #here
model_pth = "checkpoints/SR-050/best_model.pth" #here sr

if now == "vid4":
    reconstructed_videos_pth = None #here first_sr
else:
    reconstructed_videos_pth = None #here first_sr
if now == "vid4":
    test_pth = None #here
else:
    test_pth = None
    
load_ipm_model_path = None


blk_size = 16
MM = int(first_sr*blk_size*blk_size)
M = int(sr*blk_size*blk_size)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#device = 'cuda' if torch.cuda.is_available else 'cpu'
device = 'cuda'
model = CSNet(sr).to(device)
model.load_state_dict(torch.load(model_pth))
model.eval()

kernels = make_kernels(blk_size, first_sr, device) #first_srのhadamard行列
ipm = torch.load(load_ipm_model_path)
for kernel_name in ipm:
    ipm_kernel = ipm[kernel_name]
ipm_kernel = ipm_kernel[MM:M, :, :, :].to(device)  #secondのtrained hadamard行列

lis = sorted(os.listdir(test_pth))
#print(lis)
end = time.time()
psnr_av = 0
ssim_av = 0
time_av = 0

for i in range(len(lis)):
    i=0

    end = time.time()
    vid4 = os.path.join(test_pth, lis[i])
    vid4_video = sorted(os.listdir(vid4))
    vid4_num = len(vid4_video)
    print("now is processing:",vid4)
    
    inputs_imgs = torch.zeros((vid4_num, 1, 160, 160)).to(device)
    video_frames = torch.zeros((vid4_num, int(blk_size*blk_size*sr), 10, 10)).to(device)
    for j in range(vid4_num):
        input_dir = os.path.join(vid4, vid4_video[j])
        gry_img_np = open_img(input_dir)
        gry_img_tensor = torch.FloatTensor(gry_img_np)
        img_tensor = gry_img_tensor.expand(1, 1, gry_img_np.shape[0], gry_img_np.shape[1])
        inputs_imgs[j,:,:,:] = img_tensor
    
    reconstructed_videos = scipy.io.loadmat(reconstructed_videos_pth)
    if now=="vid4":
        if i == 0:
            reconstructed_frames = reconstructed_videos['calendar']
        elif i==1:
            reconstructed_frames = reconstructed_videos['city']
        elif i==2:
            reconstructed_frames = reconstructed_videos['foliage']
        elif i==3:
            reconstructed_frames = reconstructed_videos['walk']
    else:
        reconstructed_frames = reconstructed_videos['recon']
    lower_imgs = torch.zeros(vid4_num, 1, 160, 160).to(device)
    for k in range(vid4_num):
        reconstructed_frame = reconstructed_frames[:,:,:,k]
        reconstructed_frame = reconstructed_frame.reshape(1, 160, 160)
        reconst = reconstructed_frame.reshape(160, 160)
        reconstructed_frame = torch.from_numpy(reconstructed_frame[:,:,:]/255.0).float().to(device)
        #reconstructed_frame = torch.from_numpy(reconstructed_frame[:,:,:]).float().to(device)
        lower_imgs[k, :, :, :] = reconstructed_frame
    video_frames[:,:MM,:,:] = F.conv2d(inputs_imgs, kernels, stride=16, padding=0)
    video_frames[:,MM:M,:,:] = F.conv2d(lower_imgs, ipm_kernel, stride=16, padding=0)
    x = video_frames
    #print(x.shape)
    output = model(x)
    video_time = time.time() - end
    psnr_total = 0
    ssim_total = 0
    for ii in range(vid4_num):
        img1 = (inputs_imgs[ii,:,:,:]*255).to('cpu')
        img2 = (output[ii,:,:,:]*255).to('cpu')
        psnr = calc_psnr(img1, img2)
        
        img1_np = img1.to('cpu').detach().numpy().copy()
        img2_np = img2.to('cpu').detach().numpy().copy()
        img1_np = img1_np.reshape(160, 160)
        img2_np = img2_np.reshape(160, 160)
        
        ssim = structural_similarity(img1_np, img2_np)            
        psnr_total += psnr
        ssim_total += ssim
    psnr_result = psnr_total / vid4_num
    ssim_result = ssim_total / vid4_num
    print(f"average_psnr -> {psnr_result}")
    print(f"average_ssim -> {ssim_result}")
    print(f"video_time -> {video_time/vid4_num}")
    
    