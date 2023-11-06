from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import DataLoader
from utils import MakeDataset
from model import ScatteringNetwork
import argparse
import joblib

'''**********************Argument parser to take required arguments from user***************************'''

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help = "Dataset to be used", choices=("ieee"))
args = parser.parse_args()

dataset_type = args.dataset

'''***********************Prepare testing dataset****************************************'''

dataset = MakeDataset(dataset_type,"train")
test_dataloader = DataLoader(dataset, batch_size = 10, shuffle = True)

# Apply the deep scattering network
ScatNet = ScatteringNetwork()

# Load the random forest model
model = joblib.load('rdf-iter5.sav')
net_score = 0
total = 0

for (i, (imgs, labels)) in enumerate(test_dataloader):
    (N,H,W) = imgs.shape
    print(N,H,W)
    imgs = torch.reshape(imgs, (N,1,H,W))
    sn_features = ScatNet(imgs)
    preds = model.predict(sn_features)
    print(preds)
    print(labels)
    # net_score += current_score
    total += 1
    # print("Batch " + str(i) + ". Accuracy: " + str(current_score*100) + " %")
    break

print("Average accuracy = " + str(net_score) + " %")