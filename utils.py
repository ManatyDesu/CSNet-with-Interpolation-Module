import torch
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def calc_psnr(img1, img2):
    return 10. * torch.log10(255**2 / torch.mean((img1-img2)**2))

def open_img(img_pth):
    block_size = 33
    img = Image.open(img_pth)   #open the RGB image
    gry_img = img.convert('L')  #convert to gray image
    gry_img = gry_img.resize((256, 256), Image.BICUBIC)
    gry_img_np = np.array(gry_img) #convert to numpy
    #264, 264に変換する
    [row, col] = gry_img_np.shape  # 图像的 形状
    row_pad = block_size-np.mod(row,block_size)  # 求余数操作
    col_pad = block_size-np.mod(col,block_size)  # 求余数操作，用于判断需要补零的数量
    Ipad = np.concatenate((gry_img_np, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    Ipad = Ipad/255.0
    return Ipad

def make_kernels(blk_size, sr, device):
    hadamard_weights = scipy.io.loadmat('')
    hadamard_weights = hadamard_weights['x']
    hadamard_weights = hadamard_weights[:int(blk_size*blk_size*sr),:]
    hadamard_weights = torch.from_numpy(hadamard_weights).float().to(device)
    kernels = torch.zeros((int(blk_size*blk_size*sr), 1, blk_size, blk_size)).to(device)
    for i in range(int(blk_size*blk_size*sr)):
        kernel= hadamard_weights[i,:]
        kernel = kernel.reshape(blk_size, blk_size)
        kernels[i, :, :, :] = kernel
    return kernels

def make_inputs_imgs(inputs_imgs):
    num_imgs = inputs_imgs.shape[0]
    img_pth = ''
    img_lis1 = sorted(os.listdir(img_pth)) #96個のファイル
    img_num = 0
    
    for i in range(len(img_lis1)):
        if img_num == num_imgs:
            break
        img_pth2 = os.path.join(img_pth, img_lis1[i])
        img_lis2 = sorted(os.listdir(img_pth2))
        for j in range(len(img_lis2)):
            if img_num == num_imgs:
                break
            img_pth3 = os.path.join(img_pth2, img_lis2[j])
            img_lis3 = sorted(os.listdir(img_pth3))
            for k in range(len(img_lis3)):
                if img_num == num_imgs:
                    break
                print(img_num)
                fin_img_pth = os.path.join(img_pth3, img_lis3[k])
                
                #画像開く -> 画像に名前を割り当てる -> 保存する 
                gry_img_np = open_img(fin_img_pth)
                gry_img_tensor = torch.FloatTensor(gry_img_np)
                #print(gry_img_tensor.shape)
                img_tensor = gry_img_tensor.expand(1, 1, gry_img_np.shape[0], gry_img_np.shape[1])
                inputs_imgs[img_num,:,:,:] = img_tensor
                img_num += 1
    return inputs_imgs
