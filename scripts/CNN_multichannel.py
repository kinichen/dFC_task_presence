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
from utilities.train_utils import set_seed, cross_validation_control, save_model
from CNN import train_one_fold, full_retrain


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
    # Same as CNN.py

    ############## 5. Cross-Validation Run Control ######################
    best_fold_one_params = cross_validation_control(X, y, subj_label, train_config, train_one_fold, seed)
    
    ########################## 6. Optional Retrain on Full Dataset #########################
    if train_config.get('retrain_full_dataset', False):
        print("Retraining on full dataset...")
        final_model = full_retrain(dataset, best_fold_one_params)
        
        # Save final model
        paradigm = dataset_names[0].split('_')[0]  # Extract paradigm from dataset name
        final_model_path = os.path.join(model_config['output_dir'], f"model-{paradigm}-multichannel.pth")
        save_model(final_model, final_model_path)
