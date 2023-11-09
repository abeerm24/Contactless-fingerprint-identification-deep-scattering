import os
import torch.utils.data as data
import cv2
import torch
from numpy.random import randint
import random

# Create dataset of images from the given set of images 
class MakeDataset(data.Dataset):
    '''
    Class to create dataset object for pytorch. Initialization parameters:

    :param dataset_type: The dataset to be used. Currently only "ieee".
    ''' 
    def __init__(self,dataset_type, mode, classifier_type = "siamese"):
        if dataset_type=="ieee":
            '''Create the IEEE dataset object'''

            if classifier_type=="siamese":
            
                # Create gallery dataset
                PATH2gallery = "D:/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images/first_session"
                self.gallery_labels = os.listdir(PATH2gallery)

                self.gallery_paths = []

                for label in self.gallery_labels:
                    folder = os.path.join(PATH2gallery,label)
                    files = os.listdir(folder)
                    self.gallery_paths.append(os.path.join(folder,files[0]))

                # Create the required training or testing dataset
                if mode == "train":
                    PATH = "D:/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images/first_session"

                    self.same_label = [] # List to store whether two fingerprint images have the same label i.e from the same person 
                    self.path_pairs = [] # Store pair of gallery and training image paths

                    folders = os.listdir(PATH)

                    for i in range(len(self.gallery_paths)):
                        # Add 2 samples of fingerprints of the same class
                        gallery_path = self.gallery_paths[i]
                        [i1,i2] = random.sample(list(range(1,5)), 2)
                        same_label_folder = folders[i]
                        same_label_folder = os.path.join(PATH,same_label_folder)
                        same_label_files = os.listdir(same_label_folder)
                        self.path_pairs.extend([(os.path.join(same_label_folder,same_label_files[i1]),gallery_path),
                                                (os.path.join(same_label_folder,same_label_files[i2]),gallery_path)])
                        self.same_label.extend([True, True])

                        # Add 3 samples of fingerprints of different classes
                        f_list = list(range(len(self.gallery_paths)))
                        f_list.pop(i)
                        f_list = random.sample(f_list,3)
                        i_list = random.sample(list(range(1,5)), 3)
                        for pos in range(len(f_list)):
                            f = f_list[pos]
                            idx = i_list[pos]
                            folder = os.path.join(PATH, folders[f])
                            files = os.listdir(folder)
                            file = os.path.join(folder, files[idx])
                            self.path_pairs.append((file, gallery_path))
                        
                        self.same_label.extend([False,False,False])

                    self.same_label = torch.tensor(self.same_label)

                elif mode == "test":
                    PATH = "D:/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images/second_session"
                    self.same_label = []
                    self.path_pairs = []

                    folders = os.listdir(PATH)

                    for folder in folders:
                        label = folder
                        folder = os.path.join(PATH,folder)
                        files = os.listdir(folder)
                        j = randint(len(files))
                        idx_same = self.gallery_labels.index(label)
                        other_idx = random.sample(list(range(0,len(self.gallery_labels))),4)
                        other_idx.append(idx_same)
                        self.path_pairs.extend([(os.path.join(folder, files[j]), self.gallery_paths[i]) for i in other_idx])
                        self.same_label.extend([label==self.gallery_labels[i] for i in other_idx])
                    
                    self.same_label = torch.tensor(self.same_label)

                else:
                    raise Exception("Unrecognized argument. Use 'train' or 'test' ")
        
        else:
             raise Exception("Unrecognized dataset type.")
    
    def __len__(self):
         '''Return the size of the dataset'''
         return len(self.path_pairs)
         
    def gallery_size(self):
         '''Return the no. of images used in the gallery'''
         return len(self.gallery_labels)

    def __getitem__(self,index):
        same_label = float(self.same_label[index])
        (img1_path, img2_path) = self.path_pairs[index] 
        img1 =cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY) 
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)

        return (img1,img2,same_label)
