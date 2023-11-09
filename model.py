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
        self.GausBlur = GaussianBlur(kernel_size=7, sigma=4)
        
        # Define morlet wavelet convolution matrix
        l = 11 # Length of discretized morlet wavelet chosen for computation
        c = int(l/2) # Center coordinate of the constructed morlet wavelet 
        self.ds = ds
        # psi = np.fromfunction(lambda w,k,i,j: 
        #             np.exp(1j*omega0*((i-c)*(cos(pi*k/theta_div)+sin(pi*k/theta_div)) + (j-c)*(cos(pi*k/theta_div)-sin(pi*k/theta_div)))/(2*(2**w)) - ((i-c)**2+(j-c)**2)/(2*(2**w))), 
        #             (theta_div,J+1,l,l))

        # psi filter for scale 2^3
        psi_1 = np.fromfunction(lambda k,i,j: 
                    1/(sqrt(2*pi)*(2**3))*np.exp(1j*omega0*((i-c)*cos(pi*k/theta_div) - (j-c)*sin(pi*k/theta_div))/(2*(2**(2*3))) - ((i-c)**2+(j-c)**2)/(2*(2**(2*3)))), 
                    (theta_div,l,l))
        
        # psi filter for scale 2^0
        psi_2 = np.fromfunction(lambda k,i,j: 
                    1/(sqrt(2*pi)*(2**0))*np.exp(1j*omega0*((i-c)*cos(pi*k/theta_div) - (j-c)*sin(pi*k/theta_div))/(2*(2**(2*0))) - ((i-c)**2+(j-c)**2)/(2*(2**(2*0)))), 
                    (theta_div,l,l))

        # Convolution matrix for S1 layer
        self.psi1 = torch.cat((torch.tensor(psi_1),torch.tensor(psi_2)))
        self.psi1 = torch.reshape(self.psi1, (2*theta_div,1,l,l)) # Reshape the tensor into required format for convolution

        # Convolution matrix for S2 layer
        self.psi2 = tuple()
        for i in range(2*theta_div):
            self.psi2 = self.psi2 + tuple([self.psi1])
        self.psi2 = torch.cat(self.psi2, dim = 0)

    def forward(self, img):
        # Declare some parameters
        ds = self.ds
        out = self.GausBlur(img)
        
        # Extract S0 (downsample out by factor of 2^(J+1) i.e. 8) for now using downsampling of 2
        s0 = out[:,:,::ds,::ds]
        # print(s0.shape)

        img = img.to(torch.complex128)
        
        # Extract S1
        s1 = F.conv2d(img, self.psi1,padding="same")
        s1_fwd = torch.abs(s1)
        s1 = self.GausBlur(s1)
        #s1 = s1.to(torch.complex128)
        # print(s1.shape)

        # Extract S2
        #s2_layer1 = tuple()
        (N,C,H,W) = s1_fwd.shape
        # for i in range(C):
        #     s2_layer1 = s2_layer1 + tuple([F.conv2d(torch.reshape(s1[:,i,:,:],(N,1,H,W)), self.psi,padding="same")])
        s2_layer1 = F.conv2d(s1_fwd.to(torch.complex128), self.psi2, groups=C, padding="same")
        #s2_layer2 = torch.abs(torch.cat(tuple(s2_layer1),dim = 1))
        s2 = self.GausBlur(s2_layer1)
        # print(s2.shape)

        feature = torch.cat((s0,s1[:,:,::ds,::ds],s2[:,:,::ds,::ds]),dim = 1).to(torch.double).to(torch.float)
        # (N,C,H,W) = feature.shape
        # feature = torch.reshape(feature,(N,C*H*W)).to(torch.float64)
        
        return feature

class ScatteringNetwork2(torch.nn.Module):
    '''
    Class to implement the scattering network with morlet wavelet  
    '''

    def __init__(self, N = 12, omega0 = 6, theta_div = 5, ds = 8):
        super(ScatteringNetwork2, self).__init__()
        self.GausBlur = GaussianBlur(kernel_size=7, sigma=4)
        
        # Define morlet wavelet convolution matrix
        l = 11 # Length of discretized morlet wavelet chosen for computation
        c = int(l/2) # Center coordinate of the constructed morlet wavelet 
        self.ds = ds
        # psi = np.fromfunction(lambda w,k,i,j: 
        #             np.exp(1j*omega0*((i-c)*(cos(pi*k/theta_div)+sin(pi*k/theta_div)) + (j-c)*(cos(pi*k/theta_div)-sin(pi*k/theta_div)))/(2*(2**w)) - ((i-c)**2+(j-c)**2)/(2*(2**w))), 
        #             (theta_div,J+1,l,l))

        # psi filter for scale 2^3
        psi_1 = np.fromfunction(lambda k,i,j: 
                    1/(sqrt(2*pi)*(2**3))*np.exp(1j*omega0*((i-c)*cos(pi*k/theta_div) - (j-c)*sin(pi*k/theta_div))/(2*(2**(2*3))) - ((i-c)**2+(j-c)**2)/(2*(2**(2*3)))), 
                    (theta_div,l,l))
        
        # psi filter for scale 2^0
        psi_2 = np.fromfunction(lambda k,i,j: 
                    1/(sqrt(2*pi)*(2**0))*np.exp(1j*omega0*((i-c)*cos(pi*k/theta_div) - (j-c)*sin(pi*k/theta_div))/(2*(2**(2*0))) - ((i-c)**2+(j-c)**2)/(2*(2**(2*0)))), 
                    (theta_div,l,l))

        # Convolution matrix for S1 layer
        self.psi1 = torch.cat((torch.tensor(psi_1),torch.tensor(psi_2)))
        self.psi1 = torch.reshape(self.psi1, (2*theta_div,1,l,l)) # Reshape the tensor into required format for convolution

        # Convolution matrix for S2 layer
        self.psi2 = tuple()
        for i in range(2*theta_div):
            self.psi2 = self.psi2 + tuple([self.psi1])
        self.psi2 = torch.cat(self.psi2, dim = 0)

    def forward(self, img):
        # Declare some parameters
        ds = self.ds
        out = self.GausBlur(img)
        
        # Extract S0 (downsample out by factor of 2^(J+1) i.e. 8) for now using downsampling of 2
        s0 = out[:,:,::ds,::ds]
        # print(s0.shape)

        img = img.to(torch.complex128)
        
        # Extract S1
        s1 = F.conv2d(img, self.psi1,padding="same")
        s1_fwd = torch.abs(s1)
        #s1 = self.GausBlur(s1)
        #s1 = s1.to(torch.complex128)
        # print(s1.shape)

        # Extract S2
        #s2_layer1 = tuple()
        (N,C,H,W) = s1_fwd.shape
        # for i in range(C):
        #     s2_layer1 = s2_layer1 + tuple([F.conv2d(torch.reshape(s1[:,i,:,:],(N,1,H,W)), self.psi,padding="same")])
        s2_layer1 = torch.abs(F.conv2d(s1_fwd.to(torch.complex128), self.psi2, groups=C, padding="same"))
        #s2_layer2 = torch.abs(torch.cat(tuple(s2_layer1),dim = 1))
        s2 = self.GausBlur(s2_layer1)
        # print(s2.shape)

        # feature = torch.cat((s0,s1[:,:,::ds,::ds],s2[:,:,::ds,::ds]),dim = 1).to(torch.double).to(torch.float)
        feature = s2[:,:,::ds,::ds]
        # (N,C,H,W) = feature.shape
        # feature = torch.reshape(feature,(N,C*H*W)).to(torch.float64)
        
        return feature

class SiameseNetwork(torch.nn.Module):
    '''
    Class to implement the Siamese network for feature matching between fingerprint images
    '''

    def __init__(self, img_size, N = 12, omega0 = 6, theta_div = 5, ds = 4):
        super(SiameseNetwork, self).__init__()
        # Initialize scattering network
        self.ScatNet = ScatteringNetwork(N, omega0, theta_div, ds)
        
        # Define the fully connected layers of the Siamese network
        (h,w) = img_size
        #num_in_features = (h//ds)*(w//ds)*(1 + 2*theta_div + (2*theta_div)**2)
        self.cnn1 = torch.nn.Conv2d(in_channels = 111, out_channels = 50, kernel_size = 5, padding = "same")
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features = 50)
        self.cnn2 = torch.nn.Conv2d(in_channels = 50, out_channels = 10, kernel_size = 5, padding = "same")
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 10)
        self.fc_layer1 = torch.nn.Linear(in_features = 50160, out_features = 100, bias = True)
        self.fc_layer2 = torch.nn.Linear(in_features = 100, out_features = 10, bias = True)
        self.fc_layer3 = torch.nn.Linear(in_features = 10, out_features = 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, img1, img2):
        # Apply scattering transform on both the images
        x1 = self.ScatNet(img1)
        x2 = self.ScatNet(img2)

        # Apply cnns and batch normalization
        x1 = self.batchnorm1(self.cnn1(x1))    # Image 1
        x1 = self.batchnorm2(self.cnn2(x1))

        x2 = self.batchnorm1(self.cnn1(x2))    # Image 2
        x2 = self.batchnorm2(self.cnn2(x2))

        # Flatten the arrays
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)

        # Compute the difference of the extracted features
        y = torch.abs(x1 - x2)

        # Apply the fully connected layers 
        y = self.relu(self.fc_layer1(y))
        y = self.relu(self.fc_layer2(y))
        y = self.relu(self.fc_layer3(y))
        
        y = self.sigmoid(y) # Apply sigmoid activation function

        return y
    
class SiameseNetwork2(torch.nn.Module):
    '''
    Class to implement the Siamese network for feature matching between fingerprint images
    '''

    def __init__(self, ScatNet, img_size):
        super(SiameseNetwork2, self).__init__()
        # Initialize scattering network
        self.ScatNet = ScatNet
        
        # Define the fully connected layers of the Siamese network
        (h,w) = img_size
        #num_in_features = (h//ds)*(w//ds)*(1 + 2*theta_div + (2*theta_div)**2)
        # self.cnn1 = torch.nn.Conv2d(in_channels = 111, out_channels = 50, kernel_size = 5, padding = "same")
        # self.batchnorm1 = torch.nn.BatchNorm2d(num_features = 50)
        # self.cnn2 = torch.nn.Conv2d(in_channels = 50, out_channels = 10, kernel_size = 5, padding = "same")
        # self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 10)
        # self.fc_layer1 = torch.nn.Linear(in_features = 50160, out_features = 100, bias = True)
        # self.fc_layer2 = torch.nn.Linear(in_features = 100, out_features = 10, bias = True)
        # self.fc_layer3 = torch.nn.Linear(in_features = 10, out_features = 1, bias = True)
        self.fc_layer = torch.nn.Linear(in_features = 141636, out_features = 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, img1, img2):
        # Apply scattering transform on both the images
        x1 = self.ScatNet(img1)
        x2 = self.ScatNet(img2)

        # Apply cnns and batch normalization
        #x1 = self.batchnorm1(self.cnn1(x1))    # Image 1
        #x1 = self.batchnorm2(self.cnn2(x1))

        #x2 = self.batchnorm1(self.cnn1(x2))    # Image 2
        #x2 = self.batchnorm2(self.cnn2(x2))

        # Flatten the arrays
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)

        # Compute the difference of the extracted features
        y = torch.abs(x1 - x2)

        # Apply the fully connected layers 
        # y = self.relu(self.fc_layer1(y))
        # y = self.relu(self.fc_layer2(y))
        # y = self.relu(self.fc_layer3(y))
        y = self.fc_layer(y)
        
        y = self.sigmoid(y) # Apply sigmoid activation function

        return y
    
class SiameseNetwork3(torch.nn.Module):
    '''
    Class to implement the Siamese network for feature matching between fingerprint images
    '''

    def __init__(self, ScatNet, img_size):
        super(SiameseNetwork3, self).__init__()
        # Initialize scattering network
        self.ScatNet = ScatNet
        
        # Define the fully connected layers of the Siamese network
        (h,w) = img_size
        #num_in_features = (h//ds)*(w//ds)*(1 + 2*theta_div + (2*theta_div)**2)
        # self.cnn1 = torch.nn.Conv2d(in_channels = 111, out_channels = 50, kernel_size = 5, padding = "same")
        self.batchnorm = torch.nn.BatchNorm1d(127600)
        # self.cnn2 = torch.nn.Conv2d(in_channels = 50, out_channels = 10, kernel_size = 5, padding = "same")
        # self.batchnorm2 = torch.nn.BatchNorm2d(num_features = 10)
        # self.fc_layer1 = torch.nn.Linear(in_features = 50160, out_features = 100, bias = True)
        # self.fc_layer2 = torch.nn.Linear(in_features = 100, out_features = 10, bias = True)
        # self.fc_layer3 = torch.nn.Linear(in_features = 10, out_features = 1, bias = True)
        self.fc_layer = torch.nn.Linear(in_features = 127600, out_features = 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, img1, img2):
        # Apply scattering transform on both the images
        x1 = self.ScatNet(img1)
        x2 = self.ScatNet(img2)
        
        # Apply cnns and batch normalization
        #x1 = self.batchnorm1(self.cnn1(x1))    # Image 1
        #x1 = self.batchnorm2(self.cnn2(x1))

        #x2 = self.batchnorm1(self.cnn1(x2))    # Image 2
        #x2 = self.batchnorm2(self.cnn2(x2))

        # Flatten the arrays
        x1 = torch.flatten(x1, start_dim = 1)
        x2 = torch.flatten(x2, start_dim = 1)

        x1 = self.batchnorm(x1.to(torch.float32))
        x2 = self.batchnorm(x2.to(torch.float32))

        # Compute the difference of the extracted features
        y = torch.abs(x1 - x2).to(torch.float)

        # Apply the fully connected layers 
        # y = self.relu(self.fc_layer1(y))
        # y = self.relu(self.fc_layer2(y))
        # y = self.relu(self.fc_layer3(y))
        y = self.fc_layer(y)
        
        y = self.sigmoid(y) # Apply sigmoid activation function

        return y