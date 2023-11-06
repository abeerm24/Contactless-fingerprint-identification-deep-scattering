import torch
import numpy as np
from model import ScatteringNetwork
import argparse
from utils import MakeDataset
from sklearn.ensemble import RandomForestClassifier
# import tfdf.keras.RandomForestModel as RandomForestModel
import tensorflow as tf
import keras
# import pickle
import joblib
import time
from torch.utils.data import DataLoader

start = time.time()

'''**********************Argument parser to take required arguments from user***************************'''

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="Dataset to be used", choices=("ieee"))
parser.add_argument("--trees", default=100, help = "No. of trees to be used in the RDF classifier")
# parser.add_argument("--num_epochs", default = 10, help = "No. of epochs to train the RDF classifier")

args = parser.parse_args()

dataset_type = args.dataset
num_trees = args.trees
# num_epochs = args.num_epochs

'''***********************Prepare training dataset****************************************'''

dataset = MakeDataset(dataset_type,"train")

labels = []
imgs = []

# Create the vectors for training 
# for i, data in enumerate(dataset):
#     imgs.append(data["image"])
#     labels.append(data["label"])

train_dataloader = DataLoader(dataset, batch_size = 10, shuffle = True) # Split the dataset into batches of 50 images

print("Dataset prepared. Time: ", time.time() - start)

'''********************Apply scattering network on the images**********************'''

# Convert the list to a pytorch tensor
# imgs = np.array(imgs)           # Convert to numpy array
# imgs = torch.from_numpy(imgs)   # Convert to pytorch tensor
# (N,H,W) = imgs.shape
# imgs = torch.reshape(imgs,(N,1,H,W)) # Reshape the tensor to the appropriate format for deep scatterin network

# Apply the deep scattering network
ScatNet = ScatteringNetwork()
# sn_features = ScatNet(imgs)

# Initialize the random forest model
# model = RandomForestClassifier(warm_start=True, verbose = 1, n_estimators=1) 
#model = tf.estimator.RandomForestRegressor(n_estimators=100)
model = RandomForestClassifier(n_estimators = 100)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Setting warm_start = true means that successive calls to model.fit will not fit entirely new models, but add successive trees

# for epoch in range(num_epochs):
for (i, (imgs, labels)) in enumerate(train_dataloader):
    iter_start = time.time()
    (N,H,W) = imgs.shape
    imgs = torch.reshape(imgs, (N,1,H,W))
    sn_features = ScatNet(imgs).numpy()             
    sn_tensor = tf.convert_to_tensor(sn_features)   # Convert to tensorflow
    # model.fit(sn_features, labels)
    with tf.GradientTape() as tape:
        y_pred =  model(sn_tensor, training = True)
        loss_value = loss_fn(labels, y_pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    iter_end = time.time()
    if i%10==0:
        print("Iter " + str(i) + " completed. Time = ", str(iter_end-iter_start))
    #model.n_estimators += 1
    #if i==8:    break
#print("Epoch " + str(epoch) +  " completed. Time: {}" + str(time.time() - start))

#print("Feature extraction done. Time: ", time.time() - start)
print("Training complete. Time taken: ", time.time()- start)

# Save the RDF model parameters
filename = "rdf-iter6.sav"
#pickle.dump(model, open(filename, 'wb'))
joblib.dump(model, open(filename, 'wb'))

'''********************Random decision forest for classification*********************'''

#model = RandomForestClassifier(verbose = 1)


