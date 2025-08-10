import math
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import timm		# access to pretrained image classification models
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import sys
import yaml


def run(config):
    """
    Main function to run the script with the given configuration.
    """

    # Loading dataset for 1 task paradigm assessed by 1 method for all subjects (1 run)
    my_filepath = config['path']
    print(f"Loading dataset from: {my_filepath}")
    
    dFC = np.load(my_filepath, allow_pickle=True)
    dFC_dict = dFC.item() # extract the dictionary from np array

    X = dFC_dict["X"]
    y = dFC_dict["y"]
    subj_label = dFC_dict["subj_label"]
    method = dFC_dict["measure_name"]

    print(f"X Dataset loaded with shape: {X.shape}")


    # Utility functions

    def get_n_ROI(a, b, c): # solves quadratic in ax**2+bx+c=0 form.
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return ValueError("No real roots")
        root1 = (-b + math.sqrt(discriminant)) / (2*a)  # always returns a float
        root2 = (-b - math.sqrt(discriminant)) / (2*a)

        if root1 > 0:
            if root1.is_integer():
                return int(root1)
            else:
                return ValueError(f"Number of ROIs = {root1} is not an integer")
        else:
            if root2.is_integer():
                return int(root2)
            else:
                return ValueError(f"Number of ROIs = {root2} is not an integer")

    ROI = get_n_ROI(1, -1, -2 * X.shape[1])  # solves quadratic equation for number of ROIs

    def vec_to_symmetric_matrix(vec):
        mat = np.zeros((ROI, ROI))
        idx = np.triu_indices(ROI, k=1)
        mat[idx] = vec
        mat = mat + mat.T
        return mat


    # Using pre-trained model EfficientNet-B0 from torchvision

    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # Transformation pipeline
    dfc_transform = transforms.Compose([				# Input: (ROI, ROI, 3)
        transforms.ToPILImage(),                        # Convert tensor/array to PIL (Python
                                                            # Image Library) image
        transforms.Resize((224, 224)),                  # Resize to EfficientNet input size
                                                            # the 3 channels dimension is untouched
        transforms.ToTensor(),                          # Convert back to tensor (3, ROI, ROI)
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize as ImageNet
    ])

    def preprocess_dfc_matrix(dfc_matrix):
        """
        Convert a single-channel dFC matrix to a 3-channel 224x224 tensor.

        Parameters:
            dfc_matrix: np.ndarray of shape (H, W)

        Returns:
            torch.Tensor of shape (3, 224, 224) ready for CNN (EfficientNet-B0) input.
        """

        # Normalize matrix to 0–1 range (optional actually)
        dfc_norm = (dfc_matrix - np.min(dfc_matrix)) / (np.max(dfc_matrix) - np.min(dfc_matrix) + 1e-8)

        # Convert to 3 channels manually by stacking for pre-trained image model compatibility
        dfc_3channel = np.stack([dfc_norm]*3, axis=-1)  # shape (H, W, 3)

        # Apply torchvision transform
        tensor_img = dfc_transform(dfc_3channel)  # (3, 224, 224)

        return tensor_img


    # Wrap my data into a PyTorch Dataset class to make compatible with DataLoader
    # to train the data batch by batch

    class dFCDataset(Dataset):
        def __init__(self, X, y):
            """
            Parameters:
                X: 2D numpy array of shape (n_samples, num_features)
                y: 1D array-like of binary labels
            """
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            vec = self.X[idx]
            label = self.y[idx]

            # Convert to symmetric matrix on the fly to save memory
            dfc_matrix = vec_to_symmetric_matrix(vec)
            # Convert to (3, 224, 224) tensor
            tensor_img = preprocess_dfc_matrix(dfc_matrix)

            return tensor_img, float(label)


    # Create DataLoaders for training and testing
    # print("Creating DataLoaders")

    dataset = dFCDataset(X, y)	# no additional transform necessary
    train_size = int(0.8 * len(dataset))	# 8/2 train/test split
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42) 	# for reproducibility
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], 
                                            generator=generator)
    # shuffle=True at each epoch (1 complete pass of all batches of training data 
    # through NN) for better generalization of the model during training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    # Loading and modifying the pre-trained model

    model = timm.create_model('efficientnet_b0', pretrained=True) # passes imageNet weights in

    # Replace EfficientNet's classifier with a new fully connected Linear layer to 
    # directly output a single value
    # EfficientNet backbone outputs shape: (batch_size, 1280) then
    # nn.Linear(1280, 1) maps this to shape: (batch_size, 1) to compare against 
    # labels (batch_size, 1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)  # output for binary classification


    # Training loop
    # print("Beginning training...", flush=True)

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)	# move model and parameters to the right device
    # For updating model parameters by getting all trainable weights and 
    # setting learning rate = how much to change weights at each step

    # Check GPU was used (connection to remote server was established)
    # assert torch.cuda.is_available(), "CUDA not available"
    # print("Using device:", torch.cuda.get_device_name(0))
    # print("Memory allocated:", torch.cuda.memory_allocated())

    # Define/instantiate loss function: Binary Cross-Entropy Loss with logits; 
    # outputs 1 logit feature (inverse of logistic=sigmoid function to get probability)
    # Adjust the loss function for class imbalance
    num_task = (y == 1).sum()
    num_rest = (y == 0).sum()
    pos_weight = torch.tensor([num_rest / num_task], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()	# set model to training mode (dropout, batchnorm, etc. 
        # behave differently in train vs eval)
        total_loss = 0	# accumulate loss over batches for average epoch loss
        for batch_x, batch_y in train_loader:
            # batch_x.shape = (batch_size, 3, 224, 224)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().unsqueeze(1)	# unsqueeze to add 
                # a dimension for matching the output shape (batch_size, 1)

            optimizer.zero_grad()	# critical to zero the gradients before backward pass
            output = model(batch_x)	# forward pass through the model to get logit 
                # predictions as a tensor of shape (batch_size, 1)
            loss = criterion(output, batch_y)	# compute loss
            loss.backward()	# backpropagate the loss to compute gradients
            optimizer.step()	# update model parameters using the gradients
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Average Train Loss Per Batch: {total_loss/len(train_loader):.4f}")


    # Evaluation

    model.eval()	# switch model mode
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():	# saves memory since when evaluating, don't need gradients
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().unsqueeze(1)

            output = model(batch_x)	# raw logits
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()	# binary tensor based on threshold

            # move to CPU, convert to numpy and append to lists for final evaluation
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    print("Test Accuracy:", balanced_accuracy_score(all_labels, all_preds))
    print("Test AUC:", roc_auc_score(all_labels, all_probs))

