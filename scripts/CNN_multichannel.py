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

import sys
from utilities.utils import get_n_ROI, vec_to_symmetric_matrix, harmonize_TR, save_model

import random
import os

#### This script builds a pretrained CNN using 3 datasets (dFC assessment methods) 
#### as the 3 RBG image channels, testing whether differerently processed information
#### improves classification


def set_seed(seed): # for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(config, dataset_names: list):
    """
    Main function to run the script with the given configuration.
    
    Parameters:
        config: dict, configuration parameters loaded from config.yaml
        dataset_names: list of str, names of the 3 datasets to be processed
    """
    model_config = config["models"]["CNN"]
    seed = model_config["training"].get("seed", 42)
    set_seed(seed)
    
    if len(dataset_names) != 3:
        raise ValueError("This script is designed to work with exactly 3 datasets", 
                         "representing 1 task paradigm assessed by 3 methods")

    ########################## 1. Load dataset #################################
    datasets_list = []
    for dataset in dataset_names:
        my_filepath = config['datasets'][dataset]['path']
        print(f"Loading dataset from: {my_filepath}")
    
        dFC_dict = np.load(my_filepath, allow_pickle=True).item()
        datasets_list.append(dFC_dict)

        X = dFC_dict["X"]
        print(f"{dataset} - X loaded with shape: {X.shape}")

    ROI = get_n_ROI(1, -1, -2 * datasets_list[0]["X"].shape[1])  # solves quadratic equation for number of ROIs
    # assume all datasets had same number of ROIs, so we can use the first one


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

    def preprocess_dfc_matrix(dfc_matrices: list):
        """
        Convert three dFC matrices into a 3-channel 224x224 tensor.
        
        Parameters:
            dfc_matrices: list of 3 np.ndarray, each of shape (H, W)

        Returns:
            torch.Tensor of shape (3, 224, 224) ready for CNN input.
        """
        # Normalize each matrix independently to 0–1 range
        norm_mats = []
        for mat in dfc_matrices:
            mat_norm = (mat - np.min(mat)) / (np.max(mat) - np.min(mat) + 1e-8)
            norm_mats.append(mat_norm)

        # Stack to make (H, W, 3) tensor
        dfc_3channel = np.stack(norm_mats, axis=-1)

        # Apply transforms to get shape (3, 224, 224)
        tensor_img = dfc_transform(dfc_3channel)

        return tensor_img


    #################### 3. Dataset Class ################################
    # Wrap my data into a PyTorch Dataset class to make compatible with DataLoader
    # to train the data batch by batch

    class dFCDataset(Dataset):
        def __init__(self, harmonized: list):
            """
            Parameters:
                harmonized: list of three dictionaries with TR harmonized across methods
            """
            self.datasets = harmonized
            # harmonizing ensured samples are the same across datasets/methods
            self.n_samples = len(self.datasets[0]['X'])
            self.y = harmonized[0]['y']  # labels are aligned, so the same

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx): # idx of the sample to retrieve

            # Convert to symmetric matrices for a single sample
            dFC_matrices = [
                vec_to_symmetric_matrix(self.datasets[i]['X'][idx], ROI)
                for i in range(len(self.datasets))
            ]
            
            # Convert to (3, 224, 224) tensor
            tensor_img = preprocess_dfc_matrix(dFC_matrices)

            return tensor_img, float(self.y[idx])

    # Same data assessed by different dFC methods gives different number of dFC 
    # time points, so need to harmonize the samples before stacking across methods/datasets
    harmonized_datasets = harmonize_TR(datasets_list)
    print("Harmonized X.shape:", harmonized_datasets[0]['X'].shape)
    ds = dFCDataset(harmonized_datasets)  # full transformed dataset to be split in CV
    
    ############### 4. Training with CV ######################
    # Create DataLoaders for training and testing
    def train_one_fold(train_idx, test_idx, fold):
        # Create loaders
        train_loader = DataLoader(Subset(ds, train_idx),
                                  batch_size=model_config['training']['batch_size'],
                                  shuffle=True) # shuffle at each epoch for generalizability
        test_loader = DataLoader(Subset(ds, test_idx),
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
        y = ds.y
        y_train_subset = y[train_idx]
        num_task = (y_train_subset == 1).sum()
        num_rest = (y_train_subset == 0).sum()
        pos_weight = torch.tensor([num_rest / num_task], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
    
    X = ds.datasets[0]['X']  # Use the first dataset's X for stratification indices
    y = ds.y  # Labels are the same across datasets after harmonization
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
        full_loader = DataLoader(ds, batch_size=model_config['training']['batch_size'], shuffle=True)

        # Load a fresh model
        final_model = timm.create_model(model_config['name'], pretrained=True)
        in_features = final_model.classifier.in_features
        final_model.classifier = nn.Linear(in_features, 1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        final_model.to(device)

        # Loss function (class imbalance)
        y = ds.y
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
        paradigm = dataset_names[0].split('_')[0]  # Extract paradigm from dataset name
        final_model_path = os.path.join(model_config['output_dir'], f"model-{paradigm}-multichannel.pth")
        save_model(final_model, final_model_path)
