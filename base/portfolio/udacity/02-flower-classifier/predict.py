from utilities import get_args_pred, load_checkpoint, build_cat_to_name, predict
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

# Main function
if __name__ == '__main__':
    
    # Parsing arguments
    args = get_args_pred()
    
    # Build dictionary of the mapping from class indices to class names
    if args.category_names != None:
        cat_to_name = build_cat_to_name(args.category_names)
    else:
        cat_to_name = build_cat_to_name()
        
    # Set the number of most likely classes to return
    if args.top_k != None:
        top_k = args.top_k
    else:
        top_k = 5
    
    # Load model
    model = load_checkpoint(args.checkpoint)

    # Predict using the model
    probs, classes = predict(args.path_to_image, model, use_gpu=args.gpu, topk=top_k)

    # Format the predictions
    objects = []
    for class_idx in np.array(classes).flatten():
        for key, value in model.class_to_idx.items():
            if class_idx == value:
                objects.append(cat_to_name[key])

    y_pos = np.arange(len(objects))
    performance = np.array(probs).flatten()
    
    # Print out the top_k highest probability flowers
    for flower, prob in zip(objects, performance):
        print('{} with probability: {}'.format(flower, prob))
    
    
    