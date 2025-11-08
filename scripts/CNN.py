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
from utilities.train_utils import set_seed, load_dataset, \
    build_dataloaders, make_class_weight, evaluate_cnn, \
    cross_validation_control, save_model


def run(config, dataset_name: list, date_str: str):
    """
    Main function to run the script with the given configuration.
    
    Parameters:
        config (dict): configuration parameters loaded from config.yaml
        dataset_name (list): name(s) of the dataset(s) to be processed
        date_str (str): date string for logging and saving versions
    """
    model_config = config["models"]["CNN"]
    train_config = model_config["training"]
    
    seed = train_config.get("seed", 42)
    set_seed(seed)
    learning_plot = train_config["learning_plot"]
    
    if len(dataset_name) == 3:  # multichannel mode
        multichannel = True
        dataset_names = dataset_name  # list of 3 dataset names

    elif len(dataset_name) == 1:  # single channel mode
        multichannel = False

    else:
        raise ValueError("dataset_name must contain either 1 (single channel) or",
                         "3 (multichannel) dataset names.")

    ########################## 1. Load dataset(s) #################################
    if multichannel:
        # Load all three datasets into a list of dictionaries
        datasets_list = []
        for dataset_name in dataset_names:
            my_filepath = config['datasets'][dataset_name]['path']
            print(f"Loading dataset from: {my_filepath}")
            dFC_dict = np.load(my_filepath, allow_pickle=True).item()
            datasets_list.append(dFC_dict)
            print(f"{dataset_name} - X loaded with shape: {dFC_dict["X"].shape}")

        # assume all datasets had same number of ROIs, so we can use the first one
        ROI = get_n_ROI(1, -1, -2 * datasets_list[0]["X"].shape[1])
    
    else:
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

    if multichannel:
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

    else:
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
    
    if multichannel:
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
    

    else:
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
    
    
    ######################## 4. Training and Testing with CV ##############################
    def train_one_fold(train_idx, val_idx, fold, params):
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

        ####### Build model: Setup device, pre-trained model, and optimizer #######
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # To check if GPU was used (connection to remote server was established):
        # print("Using device:", torch.cuda.get_device_name(0))
        # print("Memory allocated:", torch.cuda.memory_allocated())
        
        model = timm.create_model(model_config['name'], pretrained=True)
        # Replace model's classifier with a new fully connected Linear layer to 
        # directly output a single value
        # Model backbone outputs shape: (batch_size, num_neurons) then
        # nn.Linear maps this to shape: (batch_size, 1) to compare against labels
        # Replace classifier depending on model type
        if "resnet" in model_config['name']:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
        #        nn.BatchNorm1d(in_features),
                nn.ReLU(),
                nn.Dropout(p=params['dropout']),  # % of neurons are deactivated 
                    # each forward pass to enhance robustness/regularization
                nn.Linear(in_features, 1)   # binary classification
            )

        elif "efficientnet" in model_config['name']:
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
        #        nn.BatchNorm1d(in_features),
                nn.ReLU(),
                nn.Dropout(p=params['dropout']),
                nn.Linear(in_features, 1)
            )

        else:
            raise ValueError(f"Unsupported pretrained model: {model_config['name']}")
        
        '''
        # Freeze backbone except the final 2 layers for fine-tuning (only
        # update those weights and the rest are from pretraining)
        # Reduces overfitting risk on small datasets and speeds up training
        for name, param in model.named_parameters():
            # 'fc' (resnet) or 'classifier' (efficientnet)
            if "layer4" in name or "fc" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        '''
        
        model.to(device)
        optimizer = torch.optim.AdamW(      # AdamW is Adam with weight decay applied to weights instead of gradients (decoupled)
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        # Loss function adjusted for class imbalance
        pos_weight = make_class_weight(y[train_idx], device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training loop
        if fold == 1 and learning_plot:
            inner_train_losses, val_losses = [], [] # loss trackers for figures
            
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

            if fold == 1 and learning_plot:
                # store average training loss per sample for this epoch
                inner_train_losses.append(total_loss / len(train_loader))
                # Also log validation losses
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
                        output = model(batch_x)
                        loss = criterion(output, batch_y)
                        val_loss += loss.item()
                val_losses.append(val_loss / len(val_loader))

        # Validation evaluation
        _, val_auc = evaluate_cnn(model, val_loader, device)
        
        if fold == 1 and learning_plot:
            return val_auc, inner_train_losses, val_losses

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

        ####### Build model: setup device, pre-trained model, and optimizer #######
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # To check if GPU was used (connection to remote server was established):
        # print("Using device:", torch.cuda.get_device_name(0))
        # print("Memory allocated:", torch.cuda.memory_allocated())
        
        model = timm.create_model(model_config['name'], pretrained=True)
        # Replace model's classifier with a new fully connected Linear layer to 
        # directly output a single value
        # Model backbone outputs shape: (batch_size, num_neurons) then
        # nn.Linear maps this to shape: (batch_size, 1) to compare against labels
        # Replace classifier depending on model type
        if "resnet" in model_config['name']:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
        #        nn.BatchNorm1d(in_features), # normalize inputs to final fc layer for stable activations
                nn.ReLU(),
                nn.Dropout(p=params['dropout']),  # % of neurons are deactivated
                nn.Linear(in_features, 1)
            )

        elif "efficientnet" in model_config['name']:
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
        #        nn.BatchNorm1d(in_features),
                nn.ReLU(),
                nn.Dropout(p=params['dropout']),
                nn.Linear(in_features, 1)
            )

        else:
            raise ValueError(f"Unsupported pretrained model: {model_config['name']}")
        
        '''
        # Freeze backbone except the final 2 layers for fine-tuning (only
        # update those weights and the rest are from pretraining)
        # Reduces overfitting risk on small datasets (bias variance tradeoff) and speeds up training
        for name, param in model.named_parameters():
            # 'fc' (resnet) or 'classifier' (efficientnet)
            if "layer4" in name or "fc" in name or "classifier" in name:    # fine-tune last block + head
                param.requires_grad = True
            else:
                param.requires_grad = False
        '''
        
        model.to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        
        # Loss function adjusted for class imbalance
        pos_weight = make_class_weight(y[train_idx], device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Training loop
        if fold == 1 and learning_plot:
            outer_train_losses, test_losses = [], [] # loss trackers for figures
            
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

            epoch_loss = total_loss / len(train_loader)
            if fold == 1:  # only for first fold to check trend with best params
                print(f"[Fold {fold}] Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")

            if fold == 1 and learning_plot:
                outer_train_losses.append(epoch_loss)
                # Also log test losses
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device).float().unsqueeze(1)
                        output = model(batch_x)
                        loss = criterion(output, batch_y)
                        test_loss += loss.item()
                test_losses.append(test_loss / len(test_loader))

        # Test evaluation
        acc, auc = evaluate_cnn(model, test_loader, device)

        if fold == 1 and learning_plot:
            return acc, auc, outer_train_losses, test_losses

        return acc, auc


    ################### 5. Cross-Validation Run Control ########################
    if multichannel:
        paradigm = dataset_names[0].split('_')[0]  # Extract paradigm from dataset name
        dataset_name = f"{paradigm}_multichannel-sw-tf-cap"
        
    best_fold_one_params = cross_validation_control(
        X, 
        y, 
        subj_label, 
        train_config, 
        train_one_fold,
        test_one_fold, 
        model_name="CNN", 
        dataset_name=dataset_name,
        date_str=date_str, 
        seed=seed)
    
    
    ########################## 6. Optional Retrain on Full Dataset #########################
    def full_retrain(dataset, best_params):
        full_loader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load a fresh model
        final_model = timm.create_model(model_config['name'], pretrained=True)
        
        # Replace classifier depending on model type
        if "resnet" in model_config['name']:
            in_features = final_model.fc.in_features
            final_model.fc = nn.Sequential(
                nn.Dropout(p=best_params['dropout']),  # % of neurons are deactivated 
                    # each forward pass to enhance robustness/regularization
                nn.Linear(in_features, 1)   # binary classification
            )

        elif "efficientnet" in model_config['name']:
            in_features = final_model.classifier.in_features
            final_model.classifier = nn.Sequential(
                nn.Dropout(p=best_params['dropout']),
                nn.Linear(in_features, 1)
            )

        else:
            raise ValueError(f"Unsupported pretrained model: {model_config['name']}")
        
        '''
        # Freeze backbone except the final fully connected layer for fine-tuning (only
        # update final layer's weights and the rest are from pretraining)
        # Reduces overfitting risk on small datasets and speeds up training
        for name, param in final_model.named_parameters():
            # 'fc' (resnet) or 'classifier' (efficientnet)
            if "fc" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        '''
        
        final_model.to(device)
        optimizer = torch.optim.AdamW(      # AdamW is Adam with weight decay applied to weights instead of gradients (decoupled)
            filter(lambda p: p.requires_grad, final_model.parameters()), 
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        )

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
        if multichannel:
            paradigm = dataset_names[0].split('_')[0]  # Extract paradigm from dataset name
            final_model_path = os.path.join(model_config['output_dir'], f"model-{paradigm}-multichannel.pth")
            save_model(final_model, final_model_path)
        else:
            final_model_path = os.path.join(model_config['output_dir'], f"model-{dataset_name}.pth")
            save_model(final_model, final_model_path)