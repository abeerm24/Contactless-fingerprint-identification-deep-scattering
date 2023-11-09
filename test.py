import torch
from torch.utils.data import DataLoader
from utils import MakeDataset
from model import ScatteringNetwork
import argparse
from model import SiameseNetwork2

'''**********************Argument parser to take required arguments from user***************************'''

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help = "Dataset to be used", choices=("ieee"))
args = parser.parse_args()

dataset_type = args.dataset

'''***********************Prepare testing dataset****************************************'''

dataset = MakeDataset(dataset_type,"test")
test_dataloader = DataLoader(dataset, batch_size = 10, shuffle = False)

# Apply the deep scattering network
(_, (img1, _, _)) = enumerate(test_dataloader).__next__()  # Load one image from the dataset
(_,H,W) = img1.shape                                    # Getting the image shape

# Initialize the Siamese network classifier
ScatNet = ScatteringNetwork()
siam_net = SiameseNetwork2(ScatNet,img_size=(H,W))
PATH = "siam_v2.pth"
siam_net.load_state_dict(torch.load(PATH))

correct_classify = 0
predicted_positives = 0
predicted_negatives = 0
actual_positives = 0
actual_negatives = 0
correct_positives = 0
correct_negatives = 0

for (i, (img1, img2, labels)) in enumerate(test_dataloader):
    (N,H,W) = img1.shape
    img1 = torch.reshape(img1, (N,1,H,W))
    img2 = torch.reshape(img2, (N,1,H,W))
    preds = siam_net(img1, img2)
    preds = torch.flatten(preds)
    print(preds)
    preds = (preds > 0.5).to(torch.int) # Convert the labels into True/False depending on whether predicted similarity is > or < 0.5
    labels = labels.to(torch.int)
    result = torch.eq(labels, preds)
    correct_classify += torch.sum(result)
    predicted_positives += torch.sum(preds)
    predicted_negatives += torch.sum(1-preds)
    actual_positives += torch.sum(labels)
    actual_negatives += torch.sum(1-labels)
    correct_positives += torch.sum(torch.mul(labels,preds)) 
    correct_negatives += torch.sum(torch.mul(1-labels, 1-preds))

    print("Iter "+ str(i) + " Accuracy: ", 100.*torch.sum(result).item()/N)
    print("Positives correctly detected = ", 1.*torch.sum(torch.mul(labels,preds)).item()/torch.sum(labels).item())
    print("Negatives correctly detected = ", 1.*torch.sum(torch.mul(1-labels, 1-preds)).item()/torch.sum(1-labels).item())

N = dataset.__len__()

print("Average accuracy = " + str(100.*correct_classify/N) + " %")
print("Percent of positives detected = " + str(100.*correct_positives/actual_positives) + " %")
print("Percent of negatives detected = " + str(100.*correct_negatives/actual_negatives) + " %")
print("Percent of positives correctly predicted = " + str(100.*correct_positives/predicted_positives) + " %")
print("Percent of negatives correctly predicted = " + str(100.*correct_negatives/predicted_negatives) + " %")