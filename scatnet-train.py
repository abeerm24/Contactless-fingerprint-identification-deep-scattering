import torch
import numpy as np
from model import SiameseNetwork3, ScatteringNetwork2
import argparse
from utils import MakeDataset
import time
from torch.utils.data import DataLoader
import random

start = time.time()

'''**********************Argument parser to take required arguments from user***************************'''

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Dataset to be used", choices=("ieee"))
# parser.add_argument("--trees", default=100, help = "No. of trees to be used in the RDF classifier")
parser.add_argument("--num_epochs", default = 5, type= int, help = "No. of epochs to train the RDF classifier")
parser.add_argument("--lr", default = 0.0001,type = float, help = "Learning rate to train the model")

args = parser.parse_args()

dataset_type = args.dataset
# num_trees = args.trees
num_epochs = args.num_epochs
lr = args.lr
batch_size = 10

'''***********************Prepare training dataset****************************************'''

dataset = MakeDataset(dataset_type,mode = "train", classifier_type= "siamese")

train_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True) # Split the dataset into batches of 10 images
#num_samples = 5000
#sample_indices = random.sample(range(len(train_dataloader.dataset)), num_samples)
#subset = torch.utils.data.Subset(dataset, sample_indices)
#subset_dataloader = DataLoader(subset, batch_size=batch_size, shuffle = True)

print("Dataset prepared. Time: ", time.time() - start)
print("No. of pairs: ", len(train_dataloader.dataset))

'''********************Apply DSN + Siamese network on the images**********************'''

(_, (img1, _, _)) = enumerate(train_dataloader).__next__()  # Load one image from the dataset
(_,H,W) = img1.shape                                    # Getting the image shape

# Initialize the Siamese network classifier
ScatNet = ScatteringNetwork2()
siam_net = SiameseNetwork3(ScatNet, img_size=(H,W))

# Define BCE loss function
#loss_fn = torch.nn.BCELoss()
loss_fn = torch.nn.MSELoss()

loss_vals = [] # List to store loss values

# Definie Adam optimizer
optimizer = torch.optim.Adam(siam_net.parameters(),lr = lr)
iter = 0
for epoch in range(num_epochs):
    for (i, (img1, img2, same_label)) in enumerate(train_dataloader):
        iter_start = time.time()
        (N,H,W) = img1.shape
        img1 = torch.reshape(img1, (N,1,H,W))
        img2 = torch.reshape(img2, (N,1,H,W))

        pred = torch.flatten(siam_net(img1, img2)) # Apply the siamese network to get the similarity score
        print(pred)
        print(same_label)
        # Optimizer step
        loss = loss_fn(torch.flatten(pred), same_label.to(torch.float)) # Compute loss
        optimizer.zero_grad()       # Set zero grad for optimizer
        loss.backward()             # Compute gradients and modify the model parameters
        optimizer.step()            
        
        loss_vals.append(loss.item())
        #if i%10 == 0:
        print("Epoch: " + str(epoch) + " iter: " + str(i) + " Loss: " + str(loss.item()) + " Time taken: " + str(time.time() - iter_start))
        if i%10==0:
            PATH = "siam_v3/model_" + str(iter) + ".pth"
            torch.save(siam_net.state_dict(), PATH)
    
    PATH = "siam_v3.pth"
    torch.save(siam_net.state_dict(), PATH)


print("Training complete. Time taken: ", time.time()- start)

# Save the model
PATH = "siam_v3.pth"
torch.save(siam_net.state_dict(), PATH)

file = open('siam_mse_loss-2.txt','w')
for loss_val in loss_vals:
    file.write(str(loss_val)+"\n")
file.close()

