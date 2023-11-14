import torch
import numpy as np
from model import ScatteringNetwork
from model import SiameseNetwork
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
model = SiameseNetwork(ScatNet)

# def triplet_loss(pos_dist, neg_dist, margin = 0.5):
#     loss = pos_dist - neg_dist
#     loss = torch.mean(loss)
#     return torch.max(torch.tensor([loss+margin, 0]))
#     return loss + margin

triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
optimizer = torch.optim.SGD(model.parameters(), lr=lr) # Initialize optimizer

'''********************Apply scattering+siamese network on the images**********************'''
ctrl = 0
f = open("tripple-loss-v1.txt", "a")

for epoch in range(num_epochs):
    for (i, (anchor_img,pos_img,neg_img)) in enumerate(train_dataloader):
        iter_start = time.time()
        (N,H,W) = anchor_img.shape

        anchor_img = torch.reshape(anchor_img, (N,1,H,W))
        pos_img = torch.reshape(pos_img, (N,1,H,W))
        neg_img = torch.reshape(neg_img, (N,1,H,W))

        pos_f = model(pos_img)
        neg_f = model(neg_img)
        anchor_f = model(anchor_img)

        optimizer.zero_grad()

        loss = triplet_loss(anchor_f,pos_f,neg_f)
        loss.backward()
        
        f.write(str(loss)+ "\n")
        print("Triplet loss: " + str(loss.item()))

        optimizer.step()

        iter_end = time.time()

        if i%10==0:
            filename = "siamese-models/siam_v1_" + str(ctrl) + ".pth"
            torch.save(model.state_dict, filename)
    
#print("Feature extraction done. Time: ", time.time() - start)
print("Training complete. Time taken: ", time.time()- start)

# Save the RDF model parameters
filename = "siamese-models/siam_v1_final.pth"
torch.save(model.state_dict, filename)

