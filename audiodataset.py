###################################
## LOAD AND PREPROCESS AUDIO DATA
###################################

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

mel_spectrogram = nn.Sequential(
            AT.MelSpectrogram(sample_rate=16000, 
                              n_fft=512, 
                              win_length=400,
                              hop_length=160,
                              n_mels=80),
            AT.AmplitudeToDB()
)

class CoswaraDataset(Dataset):
    """
    Code for reading the CoswaraDataset
    """

    def __init__(self, mode, n_classes=2, dataset_path='../final/Dataset/Coswara-Data/*/*/', segment_length=16000):
        

        self.CLASSES = n_classes
        self.AudioDICT = {'healthy': 0, 'positive': 1}
        self.segment_length = segment_length
        pid_list = glob.glob(dataset_path)
        
        paths = []
        labels = []
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
                paths.append(f)
                labels.append(status)
        paths = np.array(paths)
        labels = np.array(labels)
        
        n_sample = np.sum(labels == 'positive')
        h_paths = paths[labels == 'healthy']
        h_labels = labels[labels == 'healthy']
        idx_sample = np.random.choice(len(h_paths), n_sample) #class balance
        print(idx_sample)
        new_paths = np.concatenate([h_paths[idx_sample], paths[labels == 'positive']])
        new_labels = np.concatenate([h_labels[idx_sample], labels[labels == 'positive']])
        x, x_test, y, y_test = train_test_split(new_paths, new_labels, test_size=0.2, shuffle=True, stratify=new_labels, random_state=10)
        
        if (mode == 'train' or mode == 'valid'):
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y, random_state=10)
            if mode == 'train':
                self.paths, self.labels = x, y
            elif mode == 'valid':
                self.paths, self.labels = x_valid, y_valid
        elif (mode == 'test'):
            self.paths, self.labels = x_test, y_test
        _, cnts = np.unique(self.labels, return_counts=True)
        print("{} examples =  {}".format(mode, len(self.paths)), cnts)
        
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        audio = self.load_audio(self.paths[index])
#         audio = np.expand_dims(audio, 0)
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
        label_tensor = torch.tensor(self.AudioDICT[self.labels[index]], dtype=torch.long)

        return audio, label_tensor

    def load_audio(self, path):
        if not os.path.exists(path):
            print("AUDIO DOES NOT EXIST {}".format(path))
        audio, sr = librosa.load(path, sr=16000)
#         image_tensor = self.transform(image)

        return audio
