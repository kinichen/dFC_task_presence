import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from utilities.utils import get_n_ROI, vec_to_symmetric_matrix
from utilities.train_utils import set_seed, load_dataset, \
    build_dataloaders, make_class_weight, evaluate_graph, \
    cross_validation_control, save_model


def run(config, dataset_name: list):
    """
    Main function to run the script with the given configuration.
    
    Parameters:
        config (dict): configuration parameters loaded from config.yaml
        dataset_name (list): single element is the name of the dataset to be processed
    """
    model_config = config["models"]["GCN"]
    train_config = model_config["training"]
    
    seed = train_config.get("seed", 42)
    set_seed(seed)
    
    mode = model_config.get("edge_mode", "full") # graph edge connectivity mode
    k = model_config.get("k", None) # number of neighbors to keep connected to each node
    threshold = model_config.get("threshold", 0.5) # minimum edge weight to keep
    print(f"Graph connectivity mode: {mode}, (k={k} neighbours, threshold={threshold} if applicable)")


    ########################## 1. Load dataset #################################
    dataset_name = dataset_name[0]  # get the string out
    X, y, subj_label, method = load_dataset(dataset_name, config)
    ROI = get_n_ROI(1, -1, -2 * X.shape[1])
    
    
    ########################## 2. Graph Conversion #############################
    def dfc_to_graph(dfc_matrix, label):
        """
        Convert a dFC matrix into a graph with configurable connectivity.

        Args:
            dfc_matrix (ndarray): Dynamic FC matrix (ROI, ROI).
            label (int/float): Binary graph=dFC matrix label y.
        """
        num_nodes = dfc_matrix.shape[0] # equal to number of ROIs
        
        # Min–max normalization. Else, dFC values range from 1 to thousands, skewing training.
        min_val = dfc_matrix.min()
        max_val = dfc_matrix.max()
        if max_val > min_val:  # avoid division by zero if all values in matrix are equal
            dfc_matrix = (dfc_matrix - min_val) / (max_val - min_val)
        else:
            dfc_matrix = np.zeros_like(dfc_matrix)
        
        # All nodes start with identical feature
        x = torch.ones((num_nodes, 1), dtype=torch.float)
        
        # Edge construction
        edge_list = []
        if mode == "full":
            edge_list = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

        elif mode == "knn":
            if k is None:
                raise ValueError("A value for k must be specified for knn mode. See config.yaml")
            for i in range(num_nodes):
                # k+1 because the node itself will have the highest correlation (1.0)
                neighbors = torch.topk(torch.tensor(dfc_matrix[i]), k+1).indices.tolist()
                neighbors = [j for j in neighbors if j != i][:k]  # exclude self-loop
                edge_list.extend([(i, j) for j in neighbors])

        elif mode == "threshold":
            edge_list = [(i, j) for i in range(num_nodes) for j in range(num_nodes) 
                        if i != j and dfc_matrix[i, j] >= threshold]

        else:
            raise ValueError("mode must be 'full', 'knn', or 'threshold'")

        # PyG expects edge_index to be shape (2, num_edges)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weights = torch.tensor(    # only keep weights for edges that exist
            [dfc_matrix[i, j] for (i, j) in edge_list], dtype=torch.float
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weights,
            y=torch.tensor([label], dtype=torch.float),
        )


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


    ########################## 3. Build GCN Model ####################################
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
            x = global_mean_pool(x, batch)  # pooling: graph-level embeddings. 
                                            # Shape: (num_graphs, hidden_dim)
            return self.classifier(x)


    ################### 4. Training and Testing with CV ###########################
    def train_one_fold(train_idx, val_idx, params):
        '''
        Train the model for one inner fold of cross-validation. (Hyperparameter tuning)
        Parameters:
            train_idx: indices for training set
            val_idx: indices for validation set
            fold: current fold number
            params: dictionary of one combination of training parameters
        '''
        # Build dataloaders
        dataloaders = build_dataloaders(
            dataset,
            train_idx,
            test_idx=val_idx,
            gcn_mode=True,
            batch_size=params['batch_size']
        )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["test"]

        # Setup device, model, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(hidden_dim=params['hidden_dim']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # Loss function adjusted for class imbalance
        pos_weight = make_class_weight(y[train_idx], device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Training loop
        for epoch in range(params['epochs']):
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

        # Validation evaluation
        _, val_auc = evaluate_graph(model, val_loader, device)

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
            gcn_mode=True,
            batch_size=params['batch_size']
        )
        train_loader = dataloaders["train"]
        test_loader = dataloaders["test"]

        # Setup device, model, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(hidden_dim=params['hidden_dim']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        # Loss function adjusted for class imbalance
        pos_weight = make_class_weight(y[train_idx], device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Training loop
        for epoch in range(params['epochs']):
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

            if fold == 1:  # only print for first fold
                print(f"[Fold {fold}] Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

        # Test evaluation
        acc, auc = evaluate_graph(model, test_loader, device)

        return acc, auc
    

    ########################## 5. Cross Validation ##############################
    best_fold_one_params = cross_validation_control(X, y, subj_label, train_config, 
                                            train_one_fold, test_one_fold, model_name="GCN", seed=seed)
    
    
    ########################## 6. Optional Retrain on Full Dataset #########################
    if train_config.get('retrain_full_dataset', False):
        print("Retraining on full dataset...")
        full_loader = DataLoader(dataset, batch_size=best_fold_one_params['batch_size'], shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        final_model = GCN(hidden_dim=best_fold_one_params['hidden_dim']).to(device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_fold_one_params['lr'])
        pos_weight = make_class_weight(y, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(best_fold_one_params['epochs']):
            final_model.train()
            total_loss = 0
            for data in full_loader:
                data = data.to(device)
                data.edge_attr = data.edge_attr.to(device)
                data.y = data.y.to(device)
                optimizer.zero_grad()
                out = final_model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Full Dataset] Epoch {epoch+1} Train Loss: {total_loss/len(full_loader):.4f}")

        final_model_path = os.path.join(model_config['output_dir'], f"model-{dataset_name}.pth")
        save_model(final_model, final_model_path)