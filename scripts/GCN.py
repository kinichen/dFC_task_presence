import yaml
import numpy as np
import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from utilities.utils import get_n_ROI, vec_to_symmetric_matrix


def run(config):
    """
    Main function to run the script with the given configuration.
    """

    # Loading dataset for 1 task paradigm assessed by 1 method for all subjects (1 run)
    my_filepath = config['path']
    print(f"Loading dataset from: {my_filepath}")

    dFC = np.load(my_filepath, allow_pickle=True)
    dFC_dict = dFC.item() # extract the dictionary from np array

    X = dFC_dict["X"]
    y = dFC_dict["y"]
    subj_label = dFC_dict["subj_label"]
    method = dFC_dict["measure_name"]

    print(f"X Dataset loaded with shape: {X.shape}")
    
    ROI = get_n_ROI(1, -1, -2 * X.shape[1])

    def dfc_to_graph(dfc_matrix, label):
        """
        Convert a dFC matrix into a fully connected weighted graph.
        """
        num_nodes = dfc_matrix.shape[0]

        # Node features: all nodes get identical 1D feature (e.g., [[1.0], [1.0], ..., [1.0]])
        x = torch.ones((num_nodes, 1), dtype=torch.float)

        # Create edge_index for all pairs (excluding self-loops)
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t()

        # Get corresponding edge weights from the dFC matrix
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

            # Convert vector to symmetric matrix
            dfc_matrix = vec_to_symmetric_matrix(vec, ROI)

            # Convert symmetric matrix to graph data object
            data = dfc_to_graph(dfc_matrix, label)

            return data


    # Create DataLoaders for training and testing
    # print("Creating DataLoaders")
    dataset = dFCGraphDataset(X, y)
    train_size = int(0.8 * len(dataset))	# 8/2 train/test split
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42) 	# for reproducibility
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], 
                                            generator=generator)
    # shuffle=True at each epoch for better generalization during training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    # Basic 2 layer GCN model for graph = dFC matrix classification
    class GCN(torch.nn.Module):
        def __init__(self, hidden_dim=32):	# hidden_dim is the number of node features
            super(GCN, self).__init__()
            self.conv1 = GCNConv(1, hidden_dim)  # 1 input feature → hidden_dim
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, 1)  # Binary classification

        # message passing/neighbourhood aggregation and node embedding updating
        # x = node features; batch = which nodes belong to which graph in a batch
        def forward(self, x, edge_index, edge_attr, batch):
            x = self.conv1(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)	# non-linearity
            x = self.conv2(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)

            # Pooling: average node embeddings to get graph-level embedding
            x = global_mean_pool(x, batch)	# shape: (num_graphs, hidden_dim)

            out = self.classifier(x)	# 1 logit per graph; shape: (num_graphs, 1)
            return out


    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    # Training loop
    # print("Beginning training...", flush=True)
    for epoch in range(10):
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
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")


    # Evaluation loop
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
            
    print("Test Accuracy:", balanced_accuracy_score(all_labels, all_preds))
    print("Test AUC:", roc_auc_score(all_labels, all_probs))

