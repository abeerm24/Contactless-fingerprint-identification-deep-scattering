import torch
import numpy as np
from model import ScatteringNetwork
#from model import SiameseNetwork
from model import SiameseClassifier
import argparse
from utils import MakeDataset
import time
from torch.utils.data import DataLoader

start = time.time()

'''**********************Argument parser to take required arguments from user***************************'''

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Dataset to be used", choices=("ieee"))
parser.add_argument("--num_epochs", default = 10, type = int, help = "No. of epochs to train the RDF classifier")
parser.add_argument("--lr", default=1e-4, type = float, help = "Learning rate")

args = parser.parse_args()

dataset_type = args.dataset
num_epochs = args.num_epochs
lr = args.lr

'''***********************Prepare training dataset****************************************'''

dataset = MakeDataset(dataset_type,"train", model = "siamese")

train_dataloader = DataLoader(dataset, batch_size = 10, shuffle = True) # Split the dataset into batches of 50 images

print("Dataset prepared. Time: ", time.time() - start)


'''*************************Define Siamese network and triplet loss function**********************'''
# Apply the deep scattering network
ScatNet = ScatteringNetwork(theta_div = 5, ds = 4)
# model = SiameseNetwork(ScatNet)
model = SiameseClassifier(ScatNet)

# Code for implementing tripple loss with SGD optimizer 
# triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr) # Initialize optimizer

# MSE loss with SGD optimizer
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr) # Initialize optimizer

'''********************Apply scattering+siamese network on the images**********************'''
ctrl = 0
#f = open("tripple-loss-v2.txt", "a")
f = open("mse-loss-v3.txt", "a")

for epoch in range(num_epochs):
    for (i, (anchor_img,pos_img,neg_img)) in enumerate(train_dataloader):
        iter_start = time.time()
        (N,H,W) = anchor_img.shape

        anchor_img = torch.reshape(anchor_img, (N,1,H,W))
        pos_img = torch.reshape(pos_img, (N,1,H,W))
        neg_img = torch.reshape(neg_img, (N,1,H,W))

        # pos_f = model(pos_img)
        # neg_f = model(neg_img)
        # anchor_f = model(anchor_img)

        pos_preds = model(anchor_img, pos_img)
        neg_preds = model(anchor_img, neg_img)
        preds = torch.cat((pos_preds, neg_preds))
        target_labels = torch.cat((torch.ones(N),torch.zeros(N))).to(torch.float)

        optimizer.zero_grad()

        # Compute loss and gradients for triplet loss
        # loss = triplet_loss(anchor_f,pos_f,neg_f)

        # Compute loss and gradients for MSE loss
        loss = mse_loss(torch.flatten(preds), target_labels)
        
        # Compute gradients
        loss.backward()
        
        # Write triplet losses to file
        #f.write(str(loss.item())+ "\n")
        #print("Triplet loss: " + str(loss.item()))

        # Write mse loss to file
        f.write(str(loss.item())+ "\n")
        print("MSE loss: " + str(loss.item()))

        # Update weights
        optimizer.step()

        iter_end = time.time()

        if i%10==0:
            print("Time taken for iteration: ", iter_end - iter_start)
            #filename = "siamese-models/siam_v2_" + str(ctrl) + ".pth"
            filename = "siamese-classifier/siam_v3_" + str(ctrl) + ".pth"
            torch.save(model.state_dict(), filename)
            ctrl+=1
    
#print("Feature extraction done. Time: ", time.time() - start)
print("Training complete. Time taken: ", time.time()- start)

# Save the RDF model parameters
#filename = "siamese-models/siam_v2_final.pth"
filename = "siamese-classifier/siam_v3_final.pth"
torch.save(model.state_dict(), filename)

