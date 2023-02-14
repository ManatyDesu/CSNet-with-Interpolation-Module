import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# This Reshape_Concat_Adap and My_Reshape_Adap are from "https://github.com/WenxueCui/NL-CSNet-Pytorch/blob/main/lib/network.py"
class Reshape_Concat_Adap(torch.autograd.Function):
    blocksize = 0

    def __init__(self, block_size):

        Reshape_Concat_Adap.blocksize = block_size

    def forward(ctx, input_, ):
        ctx.save_for_backward(input_)

        data = torch.clone(input_.data)
        b_ = data.shape[0]
        c_ = data.shape[1]
        w_ = data.shape[2]
        h_ = data.shape[3]

        output = torch.zeros((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                              int(w_ * Reshape_Concat_Adap.blocksize), int(h_ * Reshape_Concat_Adap.blocksize))).cuda()

        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = data[:, :, i, j]
                
                data_temp = data_temp.view((b_, int(c_ / Reshape_Concat_Adap.blocksize / Reshape_Concat_Adap.blocksize),
                                            Reshape_Concat_Adap.blocksize, Reshape_Concat_Adap.blocksize))

                output[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize] += data_temp

        return output

    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        input_ = torch.clone(inp.data)
        grad_input = torch.clone(grad_output.data)

        b_ = input_.shape[0]
        c_ = input_.shape[1]
        w_ = input_.shape[2]
        h_ = input_.shape[3]

        output = torch.zeros((b_, c_, w_, h_)).cuda()
        output = output.view(b_, c_, w_, h_)
        for i in range(0, w_):
            for j in range(0, h_):
                data_temp = grad_input[:, :, i * Reshape_Concat_Adap.blocksize:(i + 1) * Reshape_Concat_Adap.blocksize,
                            j * Reshape_Concat_Adap.blocksize:(j + 1) * Reshape_Concat_Adap.blocksize]

                data_temp = data_temp.contiguous()
                data_temp = data_temp.view((b_, c_, 1, 1))
                output[:, :, i, j] += torch.squeeze(data_temp)

        return Variable(output)

def My_Reshape_Adap(input, blocksize):
    return Reshape_Concat_Adap(blocksize).apply(input)

class ResidualBlock(nn.Module):
    def __init__(self, channels, has_BN = False):
        super(ResidualBlock, self).__init__()
        self.has_BN = has_BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        if self.has_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if self.has_BN:
            residual = self.bn2(residual)

        return x + residual

class CSNet(nn.Module):
    def __init__(self, sr=0.25, blk_size=16, subrate=0.1):
        super(CSNet,self).__init__()
        self.blk_size = blk_size
        self.sr = sr      
        self.conv1 = nn.Conv2d(int(self.blk_size*self.blk_size*self.sr), self.blk_size*self.blk_size, kernel_size=1, stride=1, padding=0)

        # reconstruction network
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64, has_BN=True)
        self.block3 = ResidualBlock(64, has_BN=True)
        self.block4 = ResidualBlock(64, has_BN=True)
        self.block5 = ResidualBlock(64, has_BN=True)
        self.block6 = ResidualBlock(64, has_BN=True)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block8 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = My_Reshape_Adap(x, self.blk_size)
        
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return block8
class IPM(nn.Module):
    def __init__(self, num_channel=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, 545, kernel_size=33, stride=33, padding=0, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        
        return x