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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

import random
import os
import sys
from utilities.utils import get_n_ROI, vec_to_symmetric_matrix, save_model


def set_seed(seed): # for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(config, dataset_name: list):
    """
    Main function to run the script with the given configuration.
    
    Parameters:
        config (dict): configuration parameters loaded from config.yaml
        dataset_name (list): single element is the name of the dataset to be processed
    """
    model_config = config["models"]["CNN"]

    seed = model_config["training"].get("seed", 42)
    set_seed(seed)

    ########################## 1. Load dataset #################################
    # Loading dataset for 1 task paradigm assessed by 1 method for all subjects (1 run)
    dataset_name = dataset_name[0]  # get the string out
    my_filepath = config['datasets'][dataset_name]['path']
    print(f"Loading dataset from: {my_filepath}")
    
    dFC = np.load(my_filepath, allow_pickle=True)
    dFC_dict = dFC.item()

    X = dFC_dict["X"]
    y = dFC_dict["y"]
    subj_label = dFC_dict["subj_label"]
    method = dFC_dict["measure_name"]

    print(f"X Dataset loaded with shape: {X.shape}")

    ROI = get_n_ROI(1, -1, -2 * X.shape[1])  # solves quadratic equation for number of ROIs


    ######################### 2. Transformations ################################

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


    #################### 3. Dataset Class ################################
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
            dfc_matrix = vec_to_symmetric_matrix(vec, ROI)
            # Convert to (3, 224, 224) tensor
            tensor_img = preprocess_dfc_matrix(dfc_matrix)

            return tensor_img, float(label)

    dataset = dFCDataset(X, y)  # full transformed dataset to be split in CV
    
    ############### 4. Training with CV ######################
    # Create DataLoaders for training and testing
    def train_one_fold(train_idx, test_idx, fold):
        # Create loaders
        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=model_config['training']['batch_size'],
                                  shuffle=True) # shuffle at each epoch for generalizability
        test_loader = DataLoader(Subset(dataset, test_idx),
                                 batch_size=model_config['training']['batch_size'],
                                 shuffle=False)

        # Load pre-trained model
        model = timm.create_model(model_config['name'], pretrained=True)
        
        # Replace model's classifier with a new fully connected Linear layer to 
        # directly output a single value
        # Model backbone outputs shape: (batch_size, num_neurons) then
        # nn.Linear maps this to shape: (batch_size, 1) to compare against labels
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)  # binary classification
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # To check if GPU was used (connection to remote server was established):
        # assert torch.cuda.is_available(), "CUDA not available"
        # print("Using device:", torch.cuda.get_device_name(0))
        # print("Memory allocated:", torch.cuda.memory_allocated())

        # Loss function adjusted for class imbalance
        y_train_subset = y[train_idx]
        num_task = (y_train_subset == 1).sum()
        num_rest = (y_train_subset == 0).sum()
        pos_weight = torch.tensor([num_rest / num_task], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # penalize false negatives more

        # Optimizer
        if model_config['training']['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=model_config['training']['lr'])
        elif model_config['training']['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=model_config['training']['lr'], momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {model_config['training']['optimizer']}")

        # Training loop
        print("Beginning training loops...", flush=True)
        for epoch in range(model_config['training']['epochs']):
            model.train()   # set model to training mode (dropout, batchnorm, etc. 
                # behave differently in train vs eval)
            total_loss = 0  # accumulate loss over batches for average epoch loss
            for batch_x, batch_y in train_loader: # batch_x.shape = (batch_size, 3, 224, 224)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
                optimizer.zero_grad()   # zero the gradients accumulated before
                output = model(batch_x) # forward pass through the model to get logit predictions
                loss = criterion(output, batch_y)
                loss.backward() # backpropagate the loss to compute gradients
                optimizer.step()    # update model parameters using the gradients
                total_loss += loss.item()

            if fold == 1:  # print loss only for first fold to check trend
                print(f"[Fold {fold}] Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.3f}")

        # Evaluation
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # unsqueeze to add a dimension for matching the output shape (batch_size, 1)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
                output = model(batch_x)
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        acc = balanced_accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        print(f"[Fold {fold}] Balanced Accuracy: {acc:.3f}, AUC: {auc:.3f}")
        
        return acc, auc


    ############## 5. Cross-Validation Run Control ######################
    k = model_config['training']['k_folds']
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    acc_scores, auc_scores = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        acc, auc = train_one_fold(train_idx, test_idx, fold=fold)
        acc_scores.append(acc)
        auc_scores.append(auc)
    print(f"Average Test Balanced Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"Average Test AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    
    
    
    ###############################################################################
    ####[6. OPTIONAL - Retrain model on entire dataset and save for evaluation]####

    if model_config['training'].get('retrain_full_dataset', False):   # False if key is missing
        print("Retraining on the full dataset to produce final model...")
        
        # Full dataset loader
        full_loader = DataLoader(dataset, batch_size=model_config['training']['batch_size'], shuffle=True)

        # Load a fresh model
        final_model = timm.create_model(model_config['name'], pretrained=True)
        in_features = final_model.classifier.in_features
        final_model.classifier = nn.Linear(in_features, 1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        final_model.to(device)

        # Loss function (class imbalance)
        num_task = (y == 1).sum()
        num_rest = (y == 0).sum()
        pos_weight = torch.tensor([num_rest / num_task], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        if model_config['training']['optimizer'] == 'adam':
            optimizer = optim.Adam(final_model.parameters(), lr=model_config['training']['lr'])
        elif model_config['training']['optimizer'] == 'sgd':
            optimizer = optim.SGD(final_model.parameters(), lr=model_config['training']['lr'], momentum=0.9)

        # Training loop
        for epoch in range(model_config['training']['epochs']):
            final_model.train()
            total_loss = 0
            for batch_x, batch_y in full_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                output = final_model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Full Dataset] Epoch {epoch+1} Train Loss: {total_loss/len(full_loader):.3f}")

        # Save final model
        final_model_path = os.path.join(model_config['output_dir'], f"model-{dataset_name}.pth")
        save_model(final_model, final_model_path)