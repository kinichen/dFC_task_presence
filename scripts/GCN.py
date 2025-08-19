import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from utilities.utils import get_n_ROI, vec_to_symmetric_matrix, save_model


def set_seed(seed):
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
    model_config = config["models"]["GCN"]
    
    seed = model_config["training"].get("seed", 42)
    set_seed(seed)

    ########################## 1. Load dataset #################################
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
    ROI = get_n_ROI(1, -1, -2 * X.shape[1])


    ########################## 2. Graph Conversion #############################
    def dfc_to_graph(dfc_matrix, label):
        """
        Convert a dFC matrix into a fully connected weighted graph.
        """
        num_nodes = dfc_matrix.shape[0]
        x = torch.ones((num_nodes, 1), dtype=torch.float)  # all nodes start with identical feature
        edge_index = torch.tensor(
            [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
            dtype=torch.long
        ).t()
        edge_weights = torch.tensor(dfc_matrix[edge_index[0], edge_index[1]], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=torch.tensor([label], dtype=torch.float))

    class dFCGraphDataset(Dataset):
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
            dfc_matrix = vec_to_symmetric_matrix(vec, ROI)
            return dfc_to_graph(dfc_matrix, label)

    dataset = dFCGraphDataset(X, y)


    ########################## 3. GCN Model ####################################
    class GCN(nn.Module):   # Basic 2 layer GCN model for graph classification
        def __init__(self, hidden_dim=32):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(1, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index, edge_attr, batch):
            x = self.conv1(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = global_mean_pool(x, batch)  # pooling: graph-level embeddings. Shape: (num_graphs, hidden_dim)
            return self.classifier(x)


    ########################## 4. Training ##################################
    def train_one_fold(train_idx, test_idx, fold):
        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                                  batch_size=model_config['training']['batch_size'],
                                  shuffle=True)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx),
                                 batch_size=model_config['training']['batch_size'],
                                 shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(hidden_dim=model_config['model'].get('hidden_dim', 32)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['training']['lr'])
        
        # Class imbalance handling
        y_train_subset = y[train_idx]
        num_pos = (y_train_subset == 1).sum()
        num_neg = (y_train_subset == 0).sum()
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Training loop
        for epoch in range(model_config['training']['epochs']):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                data.edge_attr = data.edge_attr.to(device)
                data.y = data.y.to(device)

                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if fold == 1:  # print loss for first fold
                print(f"[Fold {fold}] Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

        acc = balanced_accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        print(f"[Fold {fold}] Balanced Accuracy: {acc:.3f}, AUC: {auc:.3f}")

        return acc, auc

    ########################## 5. Cross Validation ##############################
    k = model_config['training']['k_folds']
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    acc_scores, auc_scores = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        acc, auc = train_one_fold(train_idx, test_idx, fold)
        acc_scores.append(acc)
        auc_scores.append(auc)

    print(f"Average Test Balanced Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"Average Test AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")

    ########################## 6. Optional Full Retrain #########################
    if model_config['training'].get('retrain_full_dataset', False):
        print("Retraining on full dataset...")
        full_loader = DataLoader(dataset, batch_size=model_config['training']['batch_size'], shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        final_model = GCN(hidden_dim=model_config.get('hidden_dim', 32)).to(device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=model_config['training']['lr'])
        
        num_pos = (y == 1).sum()
        num_neg = (y == 0).sum()
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(model_config['training']['epochs']):
            final_model.train()
            total_loss = 0
            for data in full_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = final_model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Full Dataset] Epoch {epoch+1} Train Loss: {total_loss/len(full_loader):.3f}")

        final_model_path = os.path.join(model_config['output_dir'], f"model-{dataset_name}.pth")
        save_model(final_model, final_model_path)