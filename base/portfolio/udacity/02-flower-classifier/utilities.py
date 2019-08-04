###########################################
##### List of Utility functions for   #####
##### Training and Prediction         #####
###########################################
import argparse
from model import build_model
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

def get_args_train():
    '''
        Purpose: Function to help read in arguments when running 
        train.py and returns them in a Namespace object
        Returns: (Type: Namespace)
    '''
    parser = argparse.ArgumentParser(
        description='Trains a new network on a dataset and saves the model as a checkpoint.'
    )
    parser.add_argument('data_dir', metavar='DATA-DIRECTORY', action='store', help='Path to the data directory used for training')
    parser.add_argument('--save_dir', metavar='SAVE-DIRECTORY', action='store', help='Set directory to save the checkpoints')
    parser.add_argument('--arch', metavar='ARCHITECTURE', action='store', choices=['densenet121', 'densenet161', 'densenet201', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50'], help='Choose a pre-trained architecture from Pytorch to be used to for the model - https://pytorch.org/tutorials/vision/docs/source/models.html')
    parser.add_argument('--learning_rate', metavar='LEARNING-RATE', action='store', help='Hyperparamter: Set learning rate for model', type=float)
    parser.add_argument('--hidden_units', metavar='NUM-HIDDEN-UNITS', action='store', help='Hyperparamter: Set number of hidden units for model', type=int)
    parser.add_argument('--epochs', metavar='NUM-EPOCHS', action='store', help='Hyperparamter: Set number of epochs for model to train on', type=int)
    parser.add_argument('--gpu', action='store_true', default=False, help='Set directory to save the checkpoints')
    settings = parser.parse_args()
    
    return settings

def get_args_pred():
    '''
        Purpose: Function to help read in arguments when running 
        predict.py and returns them in a Namespace object
        Returns: (Type: Namespace)
    '''
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image along with the probability of that name. '
    )
    parser.add_argument('path_to_image', metavar='INPUT', action='store', help='Path to the image used for prediction')
    parser.add_argument('checkpoint', metavar='CHECKPOINT', action='store', help='Path to the checkpoint file from which to load model used for prediction')
    parser.add_argument('--top_k', metavar='K', action='store', help='Sets K to return top K most likely classes', type=int)
    parser.add_argument('--category_names', metavar='CATEGORY-MAPPING', action='store', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help='Set directory to save the checkpoints')
    settings = parser.parse_args()
    
    return settings

def preprocess_data(data_dir):
    '''
        Purpose: Function to help preprocess the images
        from the dataset in data_dir to be used with Pytorch
        Returns: (Type: Dictionary, Type: int, Type: torch.utils.data.DataLoader, Type: torch.utils.data.DataLoader, Type: torch.utils.data.DataLoader)
    '''
    # Directories of training, validation, and testing data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    # ImageFolder requires the directory to be
    # in this format -> ./root/cat/XXX.jpg
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # DataLoader will create a generator which we 
    # will loop through later during each epoch
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_data.class_to_idx, len(train_data.classes), trainloader, validloader, testloader

def build_cat_to_name(category_names='cat_to_name.json'):
    '''
        Purpose: Function to help build a dictionary of class indicies to class names
        Returns: (Type: Dictionary)
    '''
    # Opens json file and rebuilds dictionary mapping
    # of category indices to names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    '''
        Purpose: Function that helps to load a checkpoint.pth file
        to build a model to be used for prediction
        Returns: A model loaded from the checkpoint.pth
    '''
    # Loads our checkpoint
    checkpoint = torch.load(filepath)
    
    # Build the model from checkpoint.pth
    output_size = checkpoint['output_size']
    args = checkpoint['args']
    if args.arch != None and args.hidden_units != None:
        model = build_model(output_size, arch=args.arch, hidden_size=args.hidden_units)
    else:
        if args.arch != None: 
            model = build_model(output_size, arch=args.arch)
        elif args.hidden_units != None:
            model = build_model(output_size, hidden_size=args.hidden_units)
        else:
            model = build_model(output_size)
    
    # Load the weights and biases for this model
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Define Transforms
    process_transforms = transforms.Compose([
        
        # Resize the images where the shortest 
        # side is 256 pixels, keeping the aspect ratio
        transforms.Resize(256),
        
        # Crop out the center 224x224 portion of the image
        transforms.CenterCrop(224),
        
        # Color channels of images are typically encoded 
        # as integers 0-255, but the model expected floats 0-1
        transforms.ToTensor(),
        
        # Normalizing image
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Process the Image
    np_image = np.array(process_transforms(image))
    
    return np_image

def predict(image_path, model, use_gpu=False, topk=5):
    '''
        Purpose: Predict the class (or classes) of an image using a trained deep learning model.
        Returns: 
    '''
    # TODO: Implement the code to predict the class from an image file
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        
        image = torch.from_numpy(process_image(Image.open(image_path))).type(torch.FloatTensor).unsqueeze_(0)
        
        # Use GPU if it's available
        if use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = image.to(device)

            # Turns off the Dropout we had in the Classifier layers
            # (to prevent overfitting) so that it performs with the 
            # full network on validation images
            model = model.to(device)
            
        # Set model to evaluation mode
        model.eval()

        # Forward pass of the single image
        log_ps = model.forward(image)

        # Get probabilities from the log probabilities
        # [p = e^ln(p)]
        ps = torch.exp(log_ps)

        # Gets the most likely label prediction
        # for each image in top_class and the probability
        # in top_p
        top_p, top_class = ps.topk(topk, dim=1)
        
        return top_p[:topk], top_class[:topk]