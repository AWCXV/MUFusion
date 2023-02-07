import numpy as np
import torch
import math
import torch.nn as nn
from scipy.misc import imread, imsave, imresize
import torch.nn.functional as F
import fusion_strategy

import fusion_strategy


class TwoFusion_net(nn.Module):
    def __init__(self, input_nc=2, output_nc=1):
        super(TwoFusion_net, self).__init__()

        kernel_size = 3
        stride = 1

        in_channels = 32;
        out_channels_def = 32;
        out_channels_def2 = 64;

        # encoder
        self.conv1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, out_channels_def,kernel_size=3,stride=1,padding=0),
                nn.ReLU());
        self.conv2 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels_def,kernel_size=3,stride=1,padding=0),
                nn.ReLU());
                    
        self.conv3 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels*2, out_channels_def2,kernel_size=3,stride=2,padding=0),
                nn.ReLU());
                
        #64c,64x64
                
        self.conv4 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def2, out_channels_def2,kernel_size=3,stride=1,padding=0),
                nn.ReLU());
                
                
        self.conv5 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def2*2, out_channels_def2,kernel_size=3,stride=1,padding=0),
                nn.ReLU());
        self.conv6 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def2*3, out_channels_def2,kernel_size=3,stride=1,padding=0),
                nn.ReLU());                               

        # decoder
        self.conv66 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def2, out_channels_def2, kernel_size=3, stride=1),
                nn.ReLU());
        self.conv55 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def2*2, out_channels_def2, kernel_size=3, stride=1),
                nn.ReLU());                
        self.conv44 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def2*2, out_channels_def2, kernel_size=3, stride=1),
                nn.ReLU());
                
        self.conv33 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def2*2, out_channels_def2, kernel_size=3, stride=1),
                nn.ReLU());
        self.conv22 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def2+out_channels_def, out_channels_def, kernel_size=3, stride=1),
                nn.ReLU());
        self.conv11 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels_def*2, 1, kernel_size=3, stride=1),
                nn.Tanh());
        self.up = nn.Upsample(scale_factor =2,mode="bicubic");
  
    def encoder(self, input):
        G11 = self.conv1(input)
        G21 = self.conv2(G11);
        G31 = self.conv3(torch.cat([G11,G21],1));
        
        G41 = self.conv4(torch.cat([G31],1));
        G51 = self.conv5(torch.cat([G31,G41],1));
        G61 = self.conv6(torch.cat([G31,G41,G51],1));

        return [G11,G21,G31,G41,G51,G61]

    def decoder(self, f_en):
        G6_2 = self.conv66(torch.cat([f_en[5]],1));
        G5_2 = self.conv55(torch.cat([f_en[4],G6_2],1));
        G4_2 = self.conv44(torch.cat([f_en[3],G5_2],1));
        
        G3_2 = self.conv33(torch.cat([f_en[2],G4_2],1));
        G2_2 = self.conv22(torch.cat([f_en[1],self.up(G3_2)],1));
        G1_2 = self.conv11(torch.cat([f_en[0],G2_2],1))
        return [G1_2]
