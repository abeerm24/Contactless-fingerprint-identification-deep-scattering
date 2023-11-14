import cv2
import numpy as np
from numpy import cos, sin, sqrt
from numpy import pi
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

pi = np.pi
cos = np.cos
sin = np.sin
sqrt = np.sqrt

class ScatteringNetwork(torch.nn.Module):
    '''
    Class to implement the scattering network with morlet wavelet  
    '''

    def __init__(self, N = 12, omega0 = 6, theta_div = 5, ds = 8):
        super(ScatteringNetwork, self).__init__()
        self.GausBlur = GaussianBlur(kernel_size=7, sigma=1)
        
        # Define morlet wavelet convolution matrix
        l = 11 # Length of discretized morlet wavelet chosen for computation
        c = int(l/2) # Center coordinate of the constructed morlet wavelet 
        self.ds = ds
    
        # psi filter for scale 2^0
        psi_1 = np.fromfunction(lambda k,i,j: 
                    1/(sqrt(2*pi)*(2**0))*np.exp(1j*omega0*((i-c)*cos(pi*k/theta_div) - (j-c)*sin(pi*k/theta_div))/(2*(2**(2*0))) - ((i-c)**2+(j-c)**2)/(2*(2**(2*0)))), 
                    (theta_div,l,l))
        
        # Convolution matrix for S1 layer
        self.psi1 = torch.from_numpy(psi_1)
        self.psi1 = torch.reshape(self.psi1, (theta_div,1,l,l)) # Reshape the tensor into required format for convolution

        # Convolution matrix for S2 layer
        self.psi2 = tuple()
        for i in range(theta_div):
            self.psi2 = self.psi2 + tuple([self.psi1])
        self.psi2 = torch.cat(self.psi2, dim = 0)

    def forward(self, img):
        # Declare some parameters
        ds = self.ds
        out = self.GausBlur(img)
        
        # Extract S0 (downsample out by factor of 2^(J+1) i.e. 8) for now using downsampling of 2
        s0 = out[:,:,::ds,::ds]

        img = img.to(torch.complex128)
        
        # Extract S1
        s1 = F.conv2d(img, self.psi1,padding="same")
        s1_fwd = torch.abs(s1)
        s1 = self.GausBlur(s1)

        # Extract S2
        (N,C,H,W) = s1_fwd.shape
        s2_layer1 = torch.abs(F.conv2d(s1_fwd.to(torch.complex128), self.psi2, groups=C, padding="same"))
        s2 = self.GausBlur(s2_layer1)
        
        feature = torch.cat((s0,s1[:,:,::ds,::ds],s2[:,:,::ds,::ds]),dim = 1).to(torch.float)
        
        return feature

class SiameseNetwork(torch.nn.Module):
    '''
    Class to implement the Siamese comparision network
    '''
    def __init__(self,ScatNet,margin = 0.5):
        super(SiameseNetwork,self).__init__()
        self.scatnet = ScatNet
        self.margin = margin
        self.conv1 = torch.nn.Conv2d(31,14, (6,6), padding = "same")
        self.conv2 = torch.nn.Conv2d(14,7,(6,6),padding = "same")
        self.conv3 = torch.nn.Conv2d(14,7,(6,6),padding = 2, stride = 2)
        
        self.conv4 = torch.nn.Conv2d(14,3,(6,6), padding = "same")
        self.conv5 = torch.nn.Conv2d(3,1,(6,6), padding = "same")
        self.conv6 = torch.nn.Conv2d(3,1,(6,6),padding = 2, stride = 2)

        self.conv7 = torch.nn.Conv2d(2,1,(6,6),padding = "same")

        self.maxpool2d = torch.nn.MaxPool2d(kernel_size = (4,4), stride = 2, padding = 1)
        self.relu = torch.nn.ReLU()
    
    def compute_feature(self,img):
        img = self.scatnet(img)

        x1 = torch.nn.BatchNorm2d(31)(img)
        x1 = self.conv1(x1)
        x1 = torch.nn.BatchNorm2d(14)(x1)
        x2 = self.conv2(x1)
        x2 = self.maxpool2d(x2)
        x1 = self.conv3(x1)
        x1 = torch.cat((x1,x2),dim = 1)
        
        x1 = torch.nn.BatchNorm2d(14)(x1)
        x1 = self.conv4(x1)
        x1 = torch.nn.BatchNorm2d(3)(x1)
        x2 = self.conv5(x1)
        x2 = self.maxpool2d(x2)
        x1 = self.conv6(x1)
        x1 = torch.cat((x1,x2),dim = 1)

        x1 = torch.nn.BatchNorm2d(2)(x1)
        x1 = self.conv7(x1)
        x1 = torch.flatten(x1,start_dim = 1)

        return x1 

    def forward(self, img):
        f = self.compute_feature(img)
        
        return f
            