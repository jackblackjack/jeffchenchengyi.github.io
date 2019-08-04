###########################################
##### Building of Model given         #####
##### parameters from train.py        #####
###########################################
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import numpy as np
from PIL import Image
from pylab import rcParams
import json

def build_model(output_size, arch='resnet18', hidden_size=256):
    '''
        Purpose: Function to build the model to be used for classification,
        default will be resnet18
        Returns: (Type: Namespace)
    '''
    # Load pre-trained model (arch) from Pytorch Models
    model = getattr(models, arch)(pretrained=True)
    
    # Freezing the parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Getting Input size
    if hasattr(model, 'fc'):
        input_size = model.fc.in_features
    else:
        input_size = model.classifier[0].in_features
        
    # Building the classifier layer
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_size)),
        ('relu1', nn.ReLU()), 
        ('do1', nn.Dropout(p=0.3)),
        ('output', nn.Linear(hidden_size, output_size)),
        ('logsoftmax', nn.LogSoftmax(dim=1))]))

    # Setting classifier of the pre-trained model 
    # to this
    if hasattr(model, 'fc'):
        model.fc = classifier
    else:
        model.classifier = classifier
    
    return model