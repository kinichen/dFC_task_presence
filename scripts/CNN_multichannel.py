import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import timm		# access to pretrained image classification models
import torch.optim as optim

import os
from utilities.utils import get_n_ROI, vec_to_symmetric_matrix, harmonize_TR
from utilities.train_utils import set_seed, \
    build_dataloaders, make_class_weight, evaluate_convolutional, \
    cross_validation_control, save_model


#### This script builds a pretrained CNN using 3 datasets (dFC assessment methods) 
#### as the 3 RBG image channels, testing whether differerently processed information
#### improves classification

def run(config, dataset_names: list):
    """
    Main function to run the script with the given configuration.
    
    Parameters:
        config: dict, configuration parameters loaded from config.yaml
        dataset_names: list of str, names of the 3 datasets to be processed
    """
    model_config = config["models"]["CNN"]
    train_config = model_config["training"]
    
    seed = train_config.get("seed", 42)
    set_seed(seed)
    
    if len(dataset_names) != 3:
        raise ValueError("This script is designed to work with exactly 3 datasets", 
                         "representing 1 task paradigm assessed by 3 methods")


    ########################## 1. Load datasets #################################
    datasets_list = []
    for dataset_name in dataset_names:
        my_filepath = config['datasets'][dataset_name]['path']
        print(f"Loading dataset from: {my_filepath}")
        dFC_dict = np.load(my_filepath, allow_pickle=True).item()
        datasets_list.append(dFC_dict)
        print(f"{dataset_name} - X loaded with shape: {dFC_dict["X"].shape}")

    # assume all datasets had same number of ROIs, so we can use the first one
    ROI = get_n_ROI(1, -1, -2 * datasets_list[0]["X"].shape[1])


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

    class dFCDataset(Dataset):
        def __init__(self, harmonized: list):
            """
            Parameters:
                harmonized: list of three dictionaries with TR harmonized across methods
            """
            self.datasets = harmonized
            # harmonization guarantees that labels are all aligned, so we can take any
            self.y = harmonized[0]['y']
            self.subj_label = harmonized[0]['subj_label']

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
    dataset = dFCDataset(harmonized_datasets)  # full transformed dataset to be split in CV
    X = dataset.datasets[0]['X']  # Use the first dataset's X for stratification indices
    y = dataset.y  # Labels are the same across datasets after harmonization
    subj_label = dataset.subj_label
    print("Harmonized X.shape:", X.shape)
    
    
    ############### 4. Training with CV ######################
    def train_one_fold(train_idx, val_idx, params):
        '''
        Train the model for one inner fold of cross-validation. (Hyperparameter tuning)
        Parameters:
            train_idx: indices for training set
            val_idx: indices for validation set
            params: dictionary of one combination of training parameters
        '''
        # Build dataloaders
        dataloaders = build_dataloaders(
            dataset,
            train_idx,
            test_idx=val_idx,
            gcn_mode=False,  # CNN mode
            batch_size=params['batch_size']
        )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["test"]

        ####### Setup device, pre-trained model, and optimizer #######
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # To check if GPU was used (connection to remote server was established):
        # print("Using device:", torch.cuda.get_device_name(0))
        # print("Memory allocated:", torch.cuda.memory_allocated())
        
        model = timm.create_model(model_config['name'], pretrained=True)
        # Replace model's classifier with a new fully connected Linear layer to 
        # directly output a single value
        # Model backbone outputs shape: (batch_size, num_neurons) then
        # nn.Linear maps this to shape: (batch_size, 1) to compare against labels
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)  # binary classification
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        # Loss function adjusted for class imbalance
        pos_weight = make_class_weight(y[train_idx], device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training loop
        for epoch in range(params['epochs']):
            model.train()   # set model to training mode (dropout, batchnorm, etc. 
                            # behave differently in train vs eval)
            total_loss = 0  # accumulate loss over batches
            for batch_x, batch_y in train_loader: # batch_x.shape = (batch_size, 3, 224, 224)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
                optimizer.zero_grad()   # zero the gradients accumulated before
                output = model(batch_x) # forward pass through the model to get logit predictions
                loss = criterion(output, batch_y)
                loss.backward() # backpropagate the loss to compute gradients
                optimizer.step()    # update model parameters using the gradients
                total_loss += loss.item()

        # Validation evaluation
        _, val_auc = evaluate_convolutional(model, val_loader, device)

        return val_auc
    
    
    def test_one_fold(train_idx, test_idx, fold, params):
        '''
        Test the model for one fold of cross-validation (includes training on train+val set
        with best hyperparameters found in inner loop for that fold).
        Parameters:
            train_idx: indices for TRAIN+VAL set
            test_idx: indices for test set
            fold: current outer fold number
            params: dictionary of one combination of training parameters
        '''
        # Build dataloaders
        dataloaders = build_dataloaders(
            dataset,
            train_idx,
            test_idx,
            gcn_mode=False,  # CNN mode
            batch_size=params['batch_size']
        )
        train_loader = dataloaders["train"]
        test_loader = dataloaders["test"]

        ####### Setup device, pre-trained model, and optimizer #######
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # To check if GPU was used (connection to remote server was established):
        # print("Using device:", torch.cuda.get_device_name(0))
        # print("Memory allocated:", torch.cuda.memory_allocated())
        
        model = timm.create_model(model_config['name'], pretrained=True)
        # Replace model's classifier with a new fully connected Linear layer to 
        # directly output a single value
        # Model backbone outputs shape: (batch_size, num_neurons) then
        # nn.Linear maps this to shape: (batch_size, 1) to compare against labels
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 1)  # binary classification
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        # Loss function adjusted for class imbalance
        pos_weight = make_class_weight(y[train_idx], device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training loop
        for epoch in range(params['epochs']):
            model.train()   # set model to training mode (dropout, batchnorm, etc. 
                            # behave differently in train vs eval)
            total_loss = 0  # accumulate loss over batches
            for batch_x, batch_y in train_loader: # batch_x.shape = (batch_size, 3, 224, 224)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
                optimizer.zero_grad()   # zero the gradients accumulated before
                output = model(batch_x) # forward pass through the model to get logit predictions
                loss = criterion(output, batch_y)
                loss.backward() # backpropagate the loss to compute gradients
                optimizer.step()    # update model parameters using the gradients
                total_loss += loss.item()

            if fold == 1:  # only for first fold to check trend with best params
                print(f"[Fold {fold}] Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

        # Test evaluation
        acc, auc = evaluate_convolutional(model, test_loader, device)

        return acc, auc


    ############## 5. Cross-Validation Run Control ######################
    best_fold_one_params = cross_validation_control(X, y, subj_label, train_config, train_one_fold, test_one_fold, seed)
    
    
    ################### 6. Optional Retrain on Full Dataset ###################
    def full_retrain(dataset, best_params):
        full_loader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load a fresh model
        final_model = timm.create_model(model_config['name'], pretrained=True)
        in_features = final_model.classifier.in_features
        final_model.classifier = nn.Linear(in_features, 1)
        final_model.to(device)
        
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
        pos_weight = make_class_weight(y, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Training loop
        for epoch in range(best_params['epochs']):
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
            print(f"[Full Dataset] Epoch {epoch+1} Train Loss: {total_loss/len(full_loader):.4f}")
        return final_model
    
    if train_config.get('retrain_full_dataset', False):
        print("Retraining on full dataset...")
        final_model = full_retrain(dataset, best_fold_one_params)
        
        # Save final model
        paradigm = dataset_names[0].split('_')[0]  # Extract paradigm from dataset name
        final_model_path = os.path.join(model_config['output_dir'], f"model-{paradigm}-multichannel.pth")
        save_model(final_model, final_model_path)
