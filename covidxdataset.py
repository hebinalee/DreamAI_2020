###################################
## LOAD AND PREPROCESS IMAGE DATA
###################################

import os
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from util import read_filepaths
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

COVIDxDICT = {'NON-COVID-19': 0, 'COVID-19': 1}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, n_classes=2, dataset_path='../final/Dataset/CovidX_dataset/', dim=(224, 224)):
        

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'NON-COVID-19': 0, 'COVID-19': 1}
        testfile = '../final/Dataset/CovidX_dataset/test_split.txt'
        trainfile = '../final/Dataset/CovidX_dataset/train_split.txt'
        
        if (mode == 'train' or mode == 'valid'):
            paths, labels = read_filepaths(trainfile)
            x_train, x_valid, y_train, y_valid = train_test_split(paths, labels, test_size=0.1, stratify=labels, random_state=20)
            if mode == 'train':
                self.paths, self.labels = x_train, y_train
                self.transform = train_transformer
            elif mode == 'valid':
                self.paths, self.labels = x_valid, y_valid
                self.transform = val_transformer
        elif (mode == 'test'):
            self.paths, self.labels = read_filepaths(testfile)
            self.transform = val_transformer
        _, cnts = np.unique(self.labels, return_counts=True)
        print("{} examples =  {}".format(mode, len(self.paths)))
        
        if mode == 'valid':
            mode = 'train'
        
        self.root = str(dataset_path) + '/' + mode + '/'
        self.mode = mode
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_tensor = self.load_image(self.root + self.paths[index], self.dim, augmentation=self.mode)
        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path, dim, augmentation='test'):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)


        image_tensor = self.transform(image)

        return image_tensor
