import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import timm		# access to pretrained image classification models
import torch.optim as optim

import os
from utilities.utils import get_n_ROI, vec_to_symmetric_matrix
from utilities.train_utils import set_seed, load_dataset, \
    build_dataloaders, make_class_weight, evaluate_convolutional, \
    cross_validation_control, save_model


def run(config, dataset_name: list):
    """
    Main function to run the script with the given configuration.
    
    Parameters:
        config (dict): configuration parameters loaded from config.yaml
        dataset_name (list): single element is the name of the dataset to be processed
    """
    model_config = config["models"]["CNN"]
    train_config = model_config["training"]
    
    seed = train_config.get("seed", 42)
    set_seed(seed)

    ########################## 1. Load dataset #################################
    # Loading dataset for 1 task paradigm assessed by 1 method for all subjects (1 run)
    dataset_name = dataset_name[0]  # get the string out
    X, y, subj_label, method = load_dataset(dataset_name, config)
    ROI = get_n_ROI(1, -1, -2 * X.shape[1])  # solves quadratic equation for number of ROIs

    #################### 2. CNN Model Transformations ##########################

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
    # to train the data batch by batch for each epoch

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
    
    ######################## 4. Training with CV ##############################
    def train_one_fold(train_idx, val_idx, test_idx, fold, params):
        '''
        Train the model for one outer fold of cross-validation.
        Parameters:
            train_idx: indices for training set
            val_idx: indices for validation set
            test_idx: indices for test set
            fold: current fold number
            params: dictionary of one combination of training parameters
        '''
        # Build dataloaders
        dataloaders = build_dataloaders(
            dataset,
            train_idx,
            val_idx,
            test_idx,
            gcn_mode=False,  # CNN mode
            batch_size=params['batch_size']
        )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
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

            if fold == 1:  # only for first fold to check trend
                print(f"[Fold {fold}] Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation evaluation
        _, val_auc = evaluate_convolutional(model, val_loader, device)

        # Test evaluation
        acc, auc = evaluate_convolutional(model, test_loader, device)

        return acc, auc, val_auc, model

    ################### 5. Cross-Validation Run Control ########################
    best_fold_one_params = cross_validation_control(X, y, subj_label, train_config, train_one_fold, seed)
    
    ########################## 6. Optional Retrain on Full Dataset #########################
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
        final_model_path = os.path.join(model_config['output_dir'], f"model-{dataset_name}.pth")
        save_model(final_model, final_model_path)