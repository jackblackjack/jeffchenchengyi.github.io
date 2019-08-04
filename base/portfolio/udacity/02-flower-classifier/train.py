from utilities import get_args_train, preprocess_data
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
    args = get_args_train()
    
    # Preprocess the Dataset
    class_to_idx, output_size, trainloader, validloader, testloader = preprocess_data(args.data_dir)
    
    # Build the model
    if args.arch != None and args.hidden_units != None:
        model = build_model(output_size, arch=args.arch, hidden_size=args.hidden_units)
    else:
        if args.arch != None: 
            model = build_model(output_size, arch=args.arch)
        elif args.hidden_units != None:
            model = build_model(output_size, hidden_size=args.hidden_units)
        else:
            model = build_model(output_size)
    
    # Use GPU if it's available
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the GPU if available 
        # else CPU
        model = model.to(device)
        
    # Define Loss Function
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    # Adam uses momentum
    if args.learning_rate != None:
        learning_rate = args.learning_rate
    else:
        learning_rate = 0.003
        
    # Check if model has fully connected or classifier layer
    if hasattr(model, 'fc'):
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #################### T-R-A-I-N-I-N-G ####################
    # To track which step we are on in training
    steps = 0

    # To track how much loss we have so far
    train_losses, valid_losses = [], []

    # Train for 20 epochs (all training images will be fed into model 20 times)
    if args.epochs != None:
        epochs = args.epochs
    else:
        epochs = 20
    for epoch in range(epochs):

        # Running Loss / epoch
        running_loss = 0

        # ----------------- TRAINING ---------------------
        # Training the network with 64 images at a time
        for inputs, labels in trainloader:

            # Increment the number of steps
            steps += 1

            # Use GPU if it's available
            if args.gpu:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

            # Reset all the gradients calculated from previous backprop
            optimizer.zero_grad()

            # ----------------- FORWARD PASS ---------------------
            # Uses 64 training images as input to the VGG16 with 
            # batch norm and our classifier model, which will return
            # the log probabilities after the last logsoftmax activation
            logps = model.forward(inputs)

            # ----------------- BACKWARD PASS --------------------
            # Using the Negative Log Likelihood Loss
            # https://pytorch.org/docs/stable/nn.html#nllloss
            # to calculate our loss
            loss = criterion(logps, labels)

            # Computes all the gradients for which requires_grad = True
            # (excludes all those frozen weights
            # we used as the pre-trained model)
            loss.backward()

            # Updates all the weights in classifier using the gradients
            # computed in the previous loss.backward()
            optimizer.step()

            # Increment the running loss after one set of forward / backward pass
            running_loss += loss.item()

        # ----------------- VALIDATION --------------------- 
        # After every epoch, we will validate how well the model
        # is training
        else:

            # Initialize validation loss and accuracy
            valid_loss = 0
            accuracy = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                # Turns off the Dropout we had in the Classifier layers
                # (to prevent overfitting) so that it performs with the 
                # full network on validation images
                model.eval()

                # Loop through each 64-image batch of images
                # and calculate their loss
                for images, labels in validloader:

                    # Use GPU if it's available
                    if args.gpu:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        
                        # Move images and label tensors to the default device
                        images, labels = images.to(device), labels.to(device)

                    # Forward pass of the 64 images
                    log_ps = model.forward(images)

                    # Increment the loss in current validation session
                    valid_loss += criterion(log_ps, labels)

                    # Get probabilities from the log probabilities
                    # [p = e^ln(p)]
                    ps = torch.exp(log_ps)

                    # Gets the most likely label prediction
                    # for each image in top_class and the probability
                    # in top_p
                    top_p, top_class = ps.topk(1, dim=1)

                    # Set equals to a boolean array, True for the 
                    # predicted labels that match with actual labels,
                    # false otherwise
                    equals = top_class == labels.view(*top_class.shape)

                    # Calculate accuracy
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            # Turns back on the Dropout we had in the Classifier layers
            model.train()

            # Add the losses calculated from current epoch
            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))

            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}%".format(accuracy * 100 /len(validloader)))
            
            # ----------------- SAVING CHECKPOINT AFTER EACH EPOCH --------------------- 
            model.class_to_idx = class_to_idx
            checkpoint = {
                'class_to_idx': model.class_to_idx,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'output_size': output_size,
                'args': args,
            }
            if args.save_dir != None:
                torch.save(checkpoint, args.save_dir + 'checkpoint{}.pth'.format(epoch))
            else:
                torch.save(checkpoint, 'checkpoint{}.pth'.format(epoch))
    
    #################### T-E-S-T-I-N-G ####################
    # TODO: Do validation on the test set
    # Initialize test loss and accuracy
    test_losses = []
    test_loss = 0
    test_accuracy = 0

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():

        # Turns off the Dropout we had in the Classifier layers
        # (to prevent overfitting) so that it performs with the 
        # full network on validation images
        model.eval()

        # Loop through each 64-image batch of images
        # and calculate their loss
        for images, labels in testloader:

            # Use GPU if it's available
            if args.gpu:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Move images and label tensors to the default device
                images, labels = images.to(device), labels.to(device)

            # Forward pass of the 64 images
            log_ps = model.forward(images)

            # Increment the loss in test session
            test_loss += criterion(log_ps, labels)

            # Get probabilities from the log probabilities
            # [p = e^ln(p)]
            ps = torch.exp(log_ps)

            # Gets the most likely label prediction
            # for each image in top_class and the probability
            # in top_p
            top_p, top_class = ps.topk(1, dim=1)

            # Set equals to a boolean array, True for the 
            # predicted labels that match with actual labels,
            # false otherwise
            equals = top_class == labels.view(*top_class.shape)

            # Calculate accuracy
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))

    # Turns back on the Dropout we had in the Classifier layers
    model.train()

    # Add the losses calculated from current epoch
    test_losses.append(test_loss/len(testloader))

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}%".format(accuracy * 100 /len(testloader)))
    
    #################### S-A-V-I-N-G ####################
    model.class_to_idx = class_to_idx
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'output_size': output_size,
        'args': args,
    }
    if args.save_dir != None:
        torch.save(checkpoint, args.save_dir + 'final_checkpoint.pth')
    else:
        torch.save(checkpoint, 'final_checkpoint.pth')