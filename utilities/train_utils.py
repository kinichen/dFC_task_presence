import numpy as np
import random
import os
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from itertools import product  # for grid expansion

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from datetime import datetime


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_name, config):
    my_filepath = config['datasets'][dataset_name]['path']
    print(f"Loading dataset from: {my_filepath}")

    dFC = np.load(my_filepath, allow_pickle=True)
    dFC_dict = dFC.item()
    X = dFC_dict["X"]
    y = dFC_dict["y"]
    subj_label = dFC_dict["subj_label"]
    method = dFC_dict["measure_name"]
    print(f"X Dataset loaded with shape: {X.shape}")
    return X, y, subj_label, method


# CNN and GCN Dataloader
def build_dataloaders(dataset, train_idx, test_idx, gcn_mode, batch_size, node_level=True):
    """
    Build PyTorch dataloaders for CNN or GCN models.

    Parameters:
        dataset: torch.utils.data.Dataset
            Full dataset.
        train_idx, test_idx: array-like
            Indices for splits. test_idx is used for validation as well.
        gcn_mode: bool
            If True, build dataloaders suitable for GCNs (PyG Data objects).
        batch_size: int
            Batch size for CNNs and graph-level GCNs (small ROI graphs so want batching).
        node_level: bool, optional
            If True, use node-level GCN setup (subject graphs with timepoint nodes).
            If False, use graph-level GCN setup (each dFC matrix = one graph).

    Returns:
        dict of dataloaders or subsets.
    """
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)
    
    if gcn_mode:
        if node_level:
            # Each Data object sample = one subject (big graph of timepoints)
            # Usually process one subject at a time (no batching)
            train_loader = train_set
            test_loader = test_set
        else:
            # Each sample = one dFC matrix (small graph with num_nodes = num_ROIs)
            # Can batch multiple graphs using PyG DataLoader
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return {"train": train_loader, "test": test_loader}

    # CNN mode
    dataloaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False)
    }
    return dataloaders


def expand_param_grid(train_config):
    """
    Expand config dict with lists into a list of configs for hyperparameter tuning.
    Example:
        {"lr": [0.001, 0.01], "batch_size": [16, 32]}
        --> [{"lr": 0.001, "batch_size": 16}, {"lr": 0.001, "batch_size": 32}, ...]
    
    Parameters:
        train_config : dict. Must be config["models"][<model_name>][training].
    """
    assert train_config.get("lr") is not None, "Check that train_config is the right depth."
    keys, values = zip(*[(k, v if isinstance(v, list) else [v]) for k, v in train_config.items()])
    combos = [dict(zip(keys, combo)) for combo in product(*values)] # Cartesian product of all parameters
    return combos


def make_class_weight(y_subset, device):
    """
    Calculate class weights for imbalanced datasets.
    """
    num_pos = (y_subset == 1).sum()
    num_neg = (y_subset == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    return pos_weight   # penalize false negatives more heavily


def evaluate_gcn(model, dataloader, device, node_level):
    """
    Evaluate GCN model for either node-level or graph-level classification.
    dataloader = val/test dataloader.
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)

            if node_level:
                out = model(data.x, data.edge_index, data.edge_attr)  # no batch. Shape: (num_nodes, 1)
            else:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch) # Shape: (batch_size = num_graphs, 1)
            
            probs = torch.sigmoid(out).view(-1)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    acc = balanced_accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return acc, auc


def evaluate_cnn(model, dataloader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
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
    return acc, auc


def cross_validation_control(X, y, subj_label, train_config, train_one_fold, 
                             test_one_fold, model_name, dataset_name, date_str, seed=42):
    param_grid = expand_param_grid(train_config)
    best_fold_one_params = None # to store best params from first outer fold
    learning_plot = train_config.get('learning_plot', False)
    acc_scores, auc_scores = [], []
    total_inner_train_losses, total_val_losses = [], [] # for fold 1
    fold = 1    # iterate over number of outer folds for logging purposes
    
    outer_kf = StratifiedGroupKFold(
        n_splits=train_config['outer_folds'], shuffle=True, random_state=seed
    )
    for trainval_idx, test_idx in outer_kf.split(X, y, groups=subj_label):
        X_trainval, y_trainval, groups_trainval = (
            X[trainval_idx], y[trainval_idx], subj_label[trainval_idx]
        )
        
        inner_kf = StratifiedGroupKFold(
            n_splits=train_config['inner_folds'], shuffle=True, random_state=seed
        )
        # Hyperparameter tuning
        best_val_auc = -np.inf
        best_params = None
        val_aucs = {}
        for params in param_grid:   # every combination of hyperparameters
            param_val_aucs = [] # validation AUCs (1 per inner fold) for this param combo
            for train_idx, val_idx in inner_kf.split(X_trainval, y_trainval, groups_trainval):
                train_idx = trainval_idx[train_idx]
                val_idx = trainval_idx[val_idx]
                if fold == 1 and learning_plot:
                    val_auc, inner_train_losses, val_losses = train_one_fold(train_idx, val_idx, fold, params)
                    total_inner_train_losses.append(inner_train_losses)
                    total_val_losses.append(val_losses)
                else:
                    val_auc = train_one_fold(train_idx, val_idx, fold, params)
                
                param_val_aucs.append(val_auc)
            # since dictionary keys must be immutable, convert params dict to sorted tuple of tuples
            val_aucs[tuple(sorted(params.items()))] = np.mean(param_val_aucs)
            
        best_val_auc = max(val_aucs.values())
        best_params = dict(max(val_aucs, key=val_aucs.get)) # convert from tuple back to dict
        if fold == 1:   # for final retraining on full dataset
            best_fold_one_params = best_params

        # Retrain on train+val (80% of subjects) with best params for that outer fold, evaluate on test
        if fold == 1 and learning_plot: # save losses for fold 1
            acc, auc, outer_train_losses, test_losses = test_one_fold(
                trainval_idx, test_idx, fold, params=best_params
            )
            save_dir = os.path.join("saved_performances", model_name, "losses", date_str)
            os.makedirs(save_dir, exist_ok=True)

            np.save(os.path.join(save_dir, f"inner_train_losses_{dataset_name}.npy"),
                    np.round(np.array(total_inner_train_losses, dtype=float), decimals=5))  
                # Shape: (num_hyperparams_combos*num_inner_folds, num_epochs)
            np.save(os.path.join(save_dir, f"val_losses_{dataset_name}.npy"),
                    np.round(np.array(total_val_losses, dtype=float), decimals=5))
            np.save(os.path.join(save_dir, f"outer_train_losses_{dataset_name}.npy"),
                    np.round(np.array(outer_train_losses, dtype=float), decimals=5))
                # Shape: (num_epochs,)
            np.save(os.path.join(save_dir, f"test_losses_{dataset_name}.npy"),
                    np.round(np.array(test_losses, dtype=float), decimals=5))

        else:
            acc, auc = test_one_fold(trainval_idx, test_idx, fold, params=best_params)

        acc_scores.append(acc)
        auc_scores.append(auc)
        print(f"[Fold {fold}] Best params: {best_params}, Best Val AUC: {best_val_auc:.3f}")
        print(f"[Fold {fold}] Test Balanced Accuracy: {acc:.3f}, Test AUC: {auc:.3f}\n")
    
        fold += 1
    
    # Save metrics per outer fold and print overall metrics
    save_dir = os.path.join("saved_performances", model_name, "metrics", date_str)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"acc_{dataset_name}.npy"), 
            np.round(np.array(acc_scores, dtype=float), decimals=4))
    np.save(os.path.join(save_dir, f"auc_{dataset_name}.npy"), 
            np.round(np.array(auc_scores, dtype=float), decimals=4))
    # General reporting
    print(f"Overall Test Balanced Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"Overall Test AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}\n\n")
    
    return best_fold_one_params


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Final model saved to {path}\n\n\n")