import os
import torch.utils.data as data
import cv2
import random

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

            self.same_labels = []   # Labels of whe
            self.paths = []         # Store paths to pairs of images

            # Create the required training or testing dataset
            if mode == "train":
                PATH = "D:/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images/first_session"

                all_folders = os.listdir(PATH)

                for i in range(len(self.gallery_labels)):
                    label = self.gallery_labels[i]
                    # Add 2 images of the same person in the dataset
                    same_folder = os.path.join(PATH,label)
                    same_label_files = os.listdir(same_folder)[1:] # Discard the first image as it's used in gallery
                    same_label_files = random.sample(same_label_files,2) # Randomly sample paths to 2 images of the same person 
                    self.paths.extend([(same_label_files[0], self.gallery_paths[i]),
                                       [(same_label_files[1], self.gallery_paths[i])]]) # Append the image path pair 
                    self.same_labels.extend([1,1])

                    # Add 3 images of different label in the dataset
                    use_folders = all_folders.copy()
                    use_folders.pop(i) # Remove the same label folder
                    random_folders = random.sample(use_folders,3)
                    for random_folder in random_folders:
                        folder_path = os.path.join(PATH,random_folder)
                        random_file = random.sample(os.listdir(folder_path),1)[0] # Randomly choose one file from the folder
                        random_file = os.path.join(folder_path,random_file)
                        self.paths.append((random_file,self.gallery_paths[i]))
                        self.same_labels.append(0)

            elif mode == "test":
                PATH = "D:/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images/second_session"

            else:
                raise Exception("Unrecognized argument. Use 'train' or 'test' ")             

        else:
             raise Exception("Unrecognized dataset type.")
    
    def __len__(self):
         '''Return the size of the dataset'''
         return len(self.paths)
         
    def gallery_size(self):
         '''Return the no. of images used in the gallery'''
         return len(self.gallery_labels)

    def __getitem__(self,index):
        label = self.same_labels[index]
        img_path = self.paths[index] 
        img1 =cv2.cvtColor(cv2.imread(img_path[0]), cv2.COLOR_BGR2GRAY) 
        img2 = cv2.cvtColor(cv2.imread(img_path[1]),cv2.COLOR_BGR2GRAY)
        # return {"image" : img, "paths" : img_path}
        return (img1,img2,label)
