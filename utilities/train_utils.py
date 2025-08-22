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


def nested_cv_split(X, y, groups, outer_splits=5, inner_splits=3, random_state=42):
    """
    Perform nested cross-validation with subject grouping for splits.
    Outer CV for model evaluation, inner CV for hyperparameter tuning.

    Parameters:
        X : array-like. Features. Shape: (num_samples, num_features).
        y : array-like. Labels. Shape: (num_samples,)
        groups : array-like. Subject IDs. Shape: (num_samples,)
        outer_splits : int
            Number of outer folds (subjects will be tested exactly once).
        inner_splits : int
            Number of inner folds for hyperparameter tuning.

    Yields:
        train_idx, val_idx, test_idx : tuple of np.ndarray.
    """
    outer_kf = StratifiedGroupKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )

    for outer_trainval_idx, test_idx in outer_kf.split(X, y, groups):
        X_trainval, y_trainval, groups_trainval = (
            X[outer_trainval_idx], y[outer_trainval_idx], groups[outer_trainval_idx]
        )

        inner_kf = StratifiedGroupKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )

        for inner_train_idx, val_idx in inner_kf.split(X_trainval, y_trainval, groups_trainval):
            train_idx = outer_trainval_idx[inner_train_idx]
            val_idx = outer_trainval_idx[val_idx]
            yield train_idx, val_idx, test_idx  # return indices for each fold iteration


# CNN and GCN Dataloader
def build_dataloaders(dataset, train_idx, val_idx, test_idx, gcn_mode, batch_size):
    """
    Build PyTorch dataloaders for CNN or GCN models.

    Parameters:
        dataset : torch.utils.data.Dataset. Full dataset.
        train_idx, val_idx, test_idx : array-like. Indices for splits.
        batch_size : int
        num_workers : int
            Number of subprocesses for data loading.
        gcn_mode : bool
            If True, returns raw Subsets (no batching) for GCN, which
            handles batching itself.

    Returns:
        dict of dataloaders (or subsets if gcn_mode=True).
    """
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    if gcn_mode:
        return {"train": train_set, "val": val_set, "test": test_set}

    dataloaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False),
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
    '''Calculate class weights for imbalanced datasets.'''
    num_pos = (y_subset == 1).sum()
    num_neg = (y_subset == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    return pos_weight   # penalize false negatives more heavily


def evaluate_graph(model, dataloader, device):  # GCN evaluation
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

        acc = balanced_accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return acc, auc


def evaluate_convolutional(model, dataloader, device):  # CNN evaluation
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


def cross_validation_control(X, y, subj_label, train_config, train_one_fold, seed=42):
    param_grid = expand_param_grid(train_config)
    best_fold_one_params = None # to store best params from first fold
    acc_scores, auc_scores = [], []

    for fold, (train_idx, val_idx, test_idx) in enumerate(
            nested_cv_split(
                X, y, subj_label,
                outer_splits=train_config['outer_folds'],
                inner_splits=train_config['inner_folds'],
                random_state=seed),
            start=1):
        best_val_auc = -np.inf  # initialize best validation AUC
        best_params = None
        corresponding_test_acc = None
        # best_model_state = None # if ever want to save the best model from folds

        # Hyperparameter tuning on inner folds
        for params in param_grid:   # every combination of hyperparameters
            print(f"Training with params: {params}")
            acc, _, val_auc, model = train_one_fold(train_idx, val_idx, test_idx, fold, params)
            if val_auc > best_val_auc:
                best_val_auc = val_auc  # update
                best_params = params
                corresponding_test_acc = acc
                # best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if fold == 1:
            best_fold_one_params = best_params
        print(f"[Fold {fold}] Best params: {best_params}, Best Val AUC: {best_val_auc:.3f}, \
              Corresponding Test Bal. Accuracy: {corresponding_test_acc:.3f}")

        # Retrain on train+val (80% of subjects) with best params for that fold, evaluate on test
        trainval_idx = np.concatenate([train_idx, val_idx])
        acc, auc, _, _ = train_one_fold(trainval_idx, test_idx, test_idx, fold, params=best_params)
        acc_scores.append(acc)
        auc_scores.append(auc)

    print(f"Average Test Balanced Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"Average Test AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")
    return best_fold_one_params # for final retraining on full dataset


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Final model saved to {path}")