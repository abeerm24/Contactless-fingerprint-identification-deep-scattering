import os
import torch.utils.data as data
import cv2

# Create dataset of images from the given set of images 
class MakeDataset(data.Dataset):
    '''
    Class to create dataset object for pytorch. Initialization parameters:

    :param dataset_type: The dataset to be used. Currently only "ieee".
    ''' 
    def __init__(self,dataset_type, mode):
        if dataset_type=="ieee":
            '''Create the IEEE dataset object'''
            
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
            
            elif mode == "test":
                PATH = "D:/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images/second_session"

            else:
                raise Exception("Unrecognized argument. Use 'train' or 'test' ")

            self.labels = []
            self.paths = []

            for label in os.listdir(PATH):
                    folder = os.path.join(PATH,label)
                    files = os.listdir(folder)
                    self.labels.extend([label for _ in range(len(files))])
                    self.paths.extend([os.path.join(folder,files[i]) for i in range(len(files))])
        
        else:
             raise Exception("Unrecognized dataset type.")
    
    def __len__(self):
         '''Return the size of the dataset'''
         return len(self.paths)
         
    def gallery_size(self):
         '''Return the no. of images used in the gallery'''
         return len(self.gallery_labels)

    def __getitem__(self,index):
        label = self.labels[index]
        img_path = self.paths[index] 
        img =cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) 
        
        # return {"image" : img, "paths" : img_path}
        return (img,label)
