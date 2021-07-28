##################################
## TRAIN AND VALIDATE CLASSIFIER
##################################

import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from covidxdataset import COVIDxDataset
from metric import accuracy
from util import print_stats, print_summary, select_optimizer, MetricTracker, save_checkpoint


def train(dataset_name,device, batch_size, model, trainloader, optimizer, epoch, writer):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'nums', 'accuracy']
    train_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    train_metrics.reset()
    confusion_matrix = torch.zeros(2, 2)

    for batch_idx, input_tensors in enumerate(trainloader):
        
        optimizer.zero_grad()
        audio_data, target, image_data, _ = input_tensors
        #input_data, target = input_tensors
        
        audio_data = audio_data.to(device)
        image_data = image_data.to(device)
        #print('audio_shape:',audio_data.shape)
        target = target.to(device)
        
        if dataset_name == 'audio':
            output = model(audio_data)
        elif dataset_name == 'image':
            output = model(image_data)
        elif dataset_name == 'image_audio':
            output = model(image_data, audio_data)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        correct, nums, acc = accuracy(output, target)
        num_samples = batch_idx * batch_size + 1
        _, preds = torch.max(output, 1)
        #print('target:',target,
             #'preds:',preds)
        for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
            #print('t:',t,'p:',p)
            confusion_matrix[t.long(), p.long()] += 1
        train_metrics.update_all_metrics({'correct': correct, 'nums': nums, 'loss': loss.item(), 'accuracy': acc},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        print_stats(epoch, batch_size, num_samples, trainloader, train_metrics)
    num_samples += len(target) - 1
    print("====================================")
    print_summary(epoch, num_samples, train_metrics, mode="Training")
    print('Training_Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))
    print("====================================")
    save_checkpoint(state=model.state_dict(),path='./check_point',filename='{}_{}'.format(dataset_name, epoch))
    return train_metrics

# def save_checkpoint(state, is_best, path, filename='last'):
# confusion_matrix):

def validation(dataset_name,device, batch_size, classes, model, testloader, epoch, writer):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'nums', 'accuracy']
    val_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='val')
    val_metrics.reset()
    confusion_matrix = torch.zeros(classes, classes)
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            audio_data, target, image_data, _ = input_tensors
            #input_data, target = input_tensors
            
            audio_data = audio_data.to(device)
            image_data = image_data.to(device)            
            
            target = target.to(device)

            if dataset_name == 'audio':
                output = model(audio_data)
            elif dataset_name == 'image':
                output = model(image_data)
            elif dataset_name == 'image_audio':
                output = model(image_data, audio_data)

            loss = criterion(output, target)

            correct, nums, acc = accuracy(output, target)
            num_samples = batch_idx * batch_size + 1
            _, preds = torch.max(output, 1)
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            val_metrics.update_all_metrics({'correct': correct, 'nums': nums, 'loss': loss.item(), 'accuracy': acc},
                                           writer_step=(epoch - 1) * len(testloader) + batch_idx)
    
    num_samples += len(target) - 1
    print("====================================")
    print_summary(epoch, num_samples, val_metrics, mode="Validation")

    print('Validation_Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))
    print("====================================")
    return val_metrics, confusion_matrix
