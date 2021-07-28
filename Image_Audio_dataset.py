import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import numpy as np
from util import read_filepaths
from PIL import Image
import torchaudio
import torchaudio.transforms as AT
from torchvision import transforms
from sklearn.model_selection import train_test_split
import librosa
import json
from util import *


class Image_Audio_Dataset(Dataset):
    def __init__(self, mode, n_classes=2,
                 Image_path='../final/Dataset/CovidX_dataset/',
                 Audio_path='../final/Dataset/Coswara-Data/*/*/',
                segment_length=16000, dim=(224, 224)):
        
        self.COVIDxDICT = {'NON-COVID-19': 0, 'COVID-19': 1}
        self.AudioDICT = {'healthy': 0, 'positive': 1}
        
        self.Image_path = Image_path
        self.Audio_path = Audio_path
        
        self.CLASSES = n_classes
        self.AudioDICT = {'healthy': 0, 'positive': 1}
        self.segment_length = segment_length
        pid_list = glob.glob(Audio_path)
        self.dim = dim
        self.COVIDxDICT = {'NON-COVID-19': 0, 'COVID-19': 1}
        Image_testfile = '../final/Dataset/CovidX_dataset/test_split.txt'
        Image_trainfile = '../final/Dataset/CovidX_dataset/train_split.txt'
        
        Audio_paths = []
        Audio_labels = []
        for pid in pid_list:
            json_file = pid + 'metadata.json'
            with open(json_file) as json_file:
                json_data = json.load(json_file)
                status = json_data["covid_status"]
            if status == 'positive_mild' or status == 'positive_moderate':
                status = 'positive'
            if status != 'healthy' and status != 'positive':
                continue
            file_list = glob.glob(pid + '*.wav')
            for f in file_list:
                if 'cough' not in f:
                    continue
                Audio_paths.append(f)
                Audio_labels.append(status)
        Audio_paths = np.array(Audio_paths)
        Audio_labels = np.array(Audio_labels)
        
#         Audio_n_sample = np.sum(Audio_labels == 'positive')
#         Audio_h_paths = Audio_paths[Audio_labels == 'healthy']
#         Audio_h_labels = Audio_labels[Audio_labels == 'healthy']
        
#         idx_sample = np.random.choice(len(Audio_h_paths), Audio_n_sample) #class balance
#         Audio_new_paths = np.concatenate([Audio_h_paths[idx_sample], Audio_paths[labels == 'positive']])
#         Audio_new_labels = np.concatenate([Audio_h_labels[idx_sample], Audio_labels[labels == 'positive']])
        
        #'NON-COVID-19': 0, 'COVID-19': 1
        Image_paths, Iamge_labels = read_all_filepaths(Image_trainfile,Image_testfile)
#         Iamge_n_sample = np.sum(Image_labels == 'COVID-19')
#         Iamege_h_paths = Image_paths[Iamge_labels == 'NON-COVID-19']
#         Iamege_h_labels = Image_labels[Iamge_labels == 'NON-COVID-19']
        
#         Iamge_new_paths = np.concatenate([Iamege_h_paths[idx_sample], Image_paths[labels == 'COVID-19']])
#         Iamge_new_labels = np.concatenate([Iamege_h_labels[idx_sample], Image_labels[labels == 'COVID-19']])
        
        Audio_h_paths = Audio_paths[Audio_labels == 'healthy']
        Audio_h_lables = Audio_labels[Audio_labels == 'healthy']
        
        Image_h_paths = Image_paths[Image_labels == 'NON-COVID-19']
        Image_h_lables = Iamge_labels[Image_labels == 'NON-COVID-19']
        
        
        #####
        print(len(Audio_h_paths))#2000
        print(len(Image_h_paths))#50
        print(len(Image_paths))#15000
        print(np.unique(Iamge_labels, return_counts=True))
        print(np.sum(Iamge_labels == 'NON-COVID-19'))
        #####
        
        
        
        
        paired_paths = []
        paired_labels = []
        for idx in range(len(Audio_h_paths)):
            paired_path = {'audio':Audio_h_paths[idx], 'image':Image_h_paths[idx]} 
            
            paired_paths.append(paired_path)
            paired_labels.append(0)
        
        Audio_covid_paths = Audio_paths[Audio_labels == 'positive']
        Audio_covid_lables = Audio_paths[Audio_labels == 'positive']
        
        Image_covid_paths = Image_paths[Iamge_labels == 'COVID-19']
        Image_covid_lables = Image_lables[Iamge_labels == 'COVID-19']
        
        
        
        for idx in range(len(Audio_h_paths)):
            #paired = {'audio':Audio_covid_paths[idx], 'image':Image_covid_paths[idx], 'label':1}
            paired_path = {'audio':Audio_covid_paths[idx], 'image':Image_covid_paths[idx]}
            
            paired_paths.append(paired)
            paired_labels.append(1)
        
        paired_x, paired_x_test, paired_y, paired_y_test = train_test_split(paired_paths, paired_labels, test_size=0.2, shuffle=True, stratify=Audio_new_labels, random_state=10)
        
        
        if (mode == 'train' or mode == 'valid'):
            x_train, x_valid, y_train, y_valid = train_test_split(paired_x, paired_y, test_size=0.1, shuffle=True, stratify=y, random_state=10)
            
            if mode == 'train':
                self.audio_path = paired_x['audio']
                self.image_path = paired_x['image']
                self.lables = paired_y
            elif mode == 'valid' : 
                self.audio_path = x_valid['audio']
                self.image_path = x_valid['image']
                self.lables = y_valid
        elif (mode=='test'):
            self.audio_path = paired_x_test['audio']
            self.image_path = paired_x_test['image']
            self.lables = paired_y_test
            _,cnts = np.unique(self.labels, return_counts=True)
            print("{} examples =  {}".format(mode, len(self.paths)), cnts)
        
        self.root = str(self.Image_path) + '/' + mode + '/'
        self.mode = mode
            
            
    def __len__(self):
        return len(self.audio_path)

    def __getitem__(self, index):

        audio = self.load_audio(self.audio_path[index])
        audio = torch.from_numpy(audio)
    
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
#         mel = mel_spectrogram(audio)
#         mel = librosa.feature.melspectrogram(audio, sr=16000, n_fft=512, n_mels=80, hop_length=160)
#         mel = np.expand_dims(mel, 0)
        audio = audio.unsqueeze(0)
        audio_label_tensor = torch.tensor(self.AudioDICT[self.labels[index]], dtype=torch.long)
        
        image_tensor = self.load_image(self.root + self.image_path[index], self.dim, augmentation=self.mode)
        image_label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return audio, audio_label_tensor, image_tensor, image_label_tensor

    def load_audio(self, path):
        if not os.path.exists(path):
            print("AUDIO DOES NOT EXIST {}".format(path))
        audio, sr = librosa.load(path, sr=16000)
#         image_tensor = self.transform(image)

        return audio
    def load_image(self, img_path, dim, augmentation='test'):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)


        image_tensor = self.transform(image)

        return image_tensor 
