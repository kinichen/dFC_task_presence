import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from utilities.utils import get_n_ROI, vec_to_symmetric_matrix
from utilities.train_utils import set_seed, load_dataset, \
    build_dataloaders, make_class_weight, evaluate_gcn, \
    cross_validation_control, save_model



def run(config, dataset_name: list, date_str: str):
    """
    Main function to run the script with the given configuration.
    
    Parameters:
        config (dict): configuration parameters loaded from config.yaml
        dataset_name (list): single element is the name of the dataset to be processed
        date_str (str): date string for logging and saving purposes
    """
    model_config = config["models"]["GCN"]
    train_config = model_config["training"]
    
    seed = train_config.get("seed", 42)
    set_seed(seed)
    learning_plot = train_config["learning_plot"]
    
    k = model_config.get("k", None) # number of neighbors to keep connected to each node
    
    # node-level (nodes = time points, one graph per subject, edges = similarity between time points) 
    # or graph-level (nodes = ROI, one graph per dFC time point) classification
    node_level = model_config.get("node_level", True) 

    if node_level:
        sigma = model_config.get("sigma", None) # Gaussian rbf kernel width for edge weights. 
                            # Larger = more smoothing = more global connections.
        sigma = None if sigma in [None, "None"] else float(sigma)
        print(f"Performing NODE-LEVEL classification with k={k} neighbours and sigma={sigma}.")
    else:
        mode = model_config.get("edge_mode", "full") # graph edge connectivity mode
        threshold = model_config.get("threshold", 0.5) # minimum edge weight to keep
        print(f"Performing GRAPH-LEVEL classification; Graph connectivity mode: {mode}, (k={k} neighbours, threshold={threshold} if applicable)")


    ########################## 1. Load dataset #################################
    dataset_name = dataset_name[0]  # get the string out
    X, y, subj_label, method = load_dataset(dataset_name, config)
    ROI = get_n_ROI(1, -1, -2 * X.shape[1])
    
    
    ########################## 2. Graph Conversion #############################
    if node_level:
        def gaussian_kernel_similarity(X, sigma=None):
            """
            Compute Gaussian kernel (RBF) similarity matrix from feature matrix X.

            Args:
                X (ndarray): Shape (num_nodes, num_features)
                sigma (float): Kernel width parameter. If None, set to median of non-zero distances.

            Returns:
                A (ndarray): Symmetric similarity matrix (num_nodes, num_nodes)
            """
            D = pairwise_distances(X, metric='euclidean')   # sqrt(L2 distances) as basic edge similarity
            if sigma is None:
                sigma = np.median(D[D > 0])  # avoid 0s on diagonal = self-distances
                # print(f"Sigma not provided; using median distance = {sigma:.3f}")
            A = np.exp(-D**2 / (2 * sigma**2))
            return A


        def subj_to_graph(X_subj, y_subj):
            """
            Build a graph for one subject, where nodes = timepoints.

            Args:
                X_subj (ndarray): (T, num_features), vectorized dFCs for a single subject
                y_subj (ndarray): (T,), labels per timepoint

            Returns:
                Data: PyG graph for one subject
            """
            T = len(X_subj) # number of timepoints = number of nodes
            
            # Min–max normalization. Else, dFC values range from 1 to thousands, skewing training.
            # Note that min and max are computed per subject here since they are the same 
            # across timepoints (each ranked dFC matrix).
            min_val = X_subj.min()
            max_val = X_subj.max()
            if max_val > min_val:  # avoid division by zero if all values are equal
                X_subj = (X_subj - min_val) / (max_val - min_val)
            else:
                X_subj = np.zeros_like(X_subj)

            # 1. Compute Gaussian kernel similarity between nodes = timepoints
            A = gaussian_kernel_similarity(X_subj, sigma=sigma)

            # 2. Sparsify using k-nearest neighbors
            edge_list = []
            for i in range(T):
                neighbors = np.argsort(A[i])[-(k+1):]  # top k+1 (self included); argsort gives ascending order
                neighbors = [j for j in neighbors if j != i][-k:]   # exclude self-loop
                edge_list.extend([(i, j) for j in neighbors])
            # Make symmetric = undirected (optional)
            edge_list = list(set(edge_list + [(j, i) for (i, j) in edge_list]))

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor([A[i, j] for (i, j) in edge_list], dtype=torch.float)

            # 3. Convert to PyTorch Geometric Data object
            x = torch.tensor(X_subj, dtype=torch.float) # node features = dFC vectors at each timepoint
            y = torch.tensor(y_subj, dtype=torch.float) # Shape: (num_nodes,)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)


        class NodeLevelGraph(Dataset):
            def __init__(self, X, y, subj_label):
                """
                Build subject-level graphs for all subjects in dataset.

                Args:
                    X (ndarray): (N, num_features), all vectorized dFC matrices
                    y (ndarray): (N,), task presence labels
                    subj_label (ndarray): (N,), subject index for each sample
                """
                self.subjects = np.unique(subj_label)
                self.graphs = []

                # Pre-build all graphs, one per subject
                for subj in self.subjects:
                    subj_mask = subj_label == subj
                    X_subj = X[subj_mask]
                    y_subj = y[subj_mask]
                    subj_graph = subj_to_graph(X_subj, y_subj)
                    self.graphs.append(subj_graph)

            def __len__(self):
                return len(self.graphs)

            def __getitem__(self, idx): # get graph for one subject
                return self.graphs[idx]

        # Build PyG dataset
        dataset = NodeLevelGraph(X, y, subj_label)
    
    
    else:
        def dfc_to_graph(dfc_matrix, label):
            """
            Convert a dFC matrix into a graph with configurable connectivity.

            Args:
                dfc_matrix (ndarray): Dynamic FC matrix (ROI, ROI).
                label (int/float): Binary graph=dFC matrix label y.
                
            Returns:
                Data: PyG graph for one time point.
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
            y = torch.tensor([label], dtype=torch.float)    # Shape: (1,)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)


        class GraphLevelGraph(Dataset):
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

        dataset = GraphLevelGraph(X, y)


    ########################## 3. Build 2-layer GCN Model ####################################
    class NodeLevelGCN(nn.Module):
        def __init__(self, in_channels, hidden_dim=32, dropout=0.5):
            super().__init__()  # original x shape: (num_nodes, num_features), so in_channels = num_features
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, 1)
            # self.dropout = dropout

        def forward(self, x, edge_index, edge_attr):    # no batch here; one graph per subject
            x = self.conv1(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after activation

            x = self.conv2(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
            return self.classifier(x)


    class GraphLevelGCN(nn.Module):
        def __init__(self, hidden_dim=32, dropout=0.5):
            super().__init__()
            self.conv1 = GCNConv(1, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, 1)
            # self.dropout = dropout

        def forward(self, x, edge_index, edge_attr, batch): # batch: vector mapping each node to its graph in the batch
            x = self.conv1(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv2(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)

            x = global_mean_pool(x, batch)  # pooling: graph-level embeddings. 
                                            # Shape: (num_graphs, hidden_dim)
            return self.classifier(x)


    ################### 4. Training and Testing with CV ###########################
    def train_one_fold(train_idx, val_idx, fold, params):
        '''
        Train the model for one inner fold of cross-validation. (Hyperparameter tuning)
        Parameters:
            train_idx: indices for training set based on samples/rows of X
            val_idx: indices for validation set based on samples/rows of X
            fold: current fold number
            params: dictionary of one combination of training parameters
        '''
        if node_level:  # Critical to do mapping to unique subject graphs to avoid index errors
            # Map sample indices to subject indices (not subj id strings)
            train_subj = np.unique(subj_label[train_idx])
            val_subj = np.unique(subj_label[val_idx])

            # Now map those subjects back to integer positions in dataset.graphs
            subj_train_idx = np.array([i for i, subj in enumerate(dataset.subjects) if subj in train_subj])
            subj_val_idx = np.array([i for i, subj in enumerate(dataset.subjects) if subj in val_subj])
            
            # Build dataloaders
            dataloaders = build_dataloaders(
                dataset,
                train_idx=subj_train_idx,
                test_idx=subj_val_idx,
                gcn_mode=True,
                batch_size=params['batch_size'],
                node_level=node_level
            )

        else:
            dataloaders = build_dataloaders(
                dataset,
                train_idx,
                test_idx=val_idx,
                gcn_mode=True,
                batch_size=params['batch_size'],
                node_level=node_level
            )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["test"]
        

        # Setup device, model, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if node_level:
            model = NodeLevelGCN(in_channels=X.shape[1], hidden_dim=params["hidden_dim"], dropout=params["dropout"]).to(device)
        else:
            model = GraphLevelGCN(hidden_dim=params["hidden_dim"], dropout=params["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        # Loss function adjusted for class imbalance
        pos_weight = make_class_weight(y[train_idx], device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        if fold == 1 and learning_plot:
            inner_train_losses, val_losses = [], []

        # Training loop
        for epoch in range(params['epochs']):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                data.edge_attr = data.edge_attr.to(device)
                data.y = data.y.to(device)
                
                optimizer.zero_grad()
                if node_level:
                    out = model(data.x, data.edge_index, data.edge_attr)
                else:
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y.view(-1, 1).float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            
            if fold == 1 and learning_plot:
                inner_train_losses.append(total_loss / len(train_loader))
                # Validation losses
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        if node_level:
                            out = model(data.x, data.edge_index, data.edge_attr)
                        else:
                            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                        loss = criterion(out, data.y.view(-1, 1).float())
                        val_loss += loss.item()
                val_losses.append(val_loss / len(val_loader))


        _, val_auc = evaluate_gcn(model, val_loader, device, node_level=node_level)

        if fold == 1 and learning_plot:
            return val_auc, inner_train_losses, val_losses
        return val_auc


    def test_one_fold(train_idx, test_idx, fold, params):
        '''
        Test the model for one fold of cross-validation (includes training on train+val set
        with best hyperparameters found in inner loop for that fold).
        Parameters:
            train_idx: indices for TRAIN+VAL set based on rows of X
            test_idx: indices for test set based on rows of X
            fold: current outer fold number
            params: dictionary of one combination of training parameters
        '''
        if node_level:
            # Map sample indices to subject indices (not subj id strings)
            train_subj = np.unique(subj_label[train_idx])
            test_subj = np.unique(subj_label[test_idx])

            # Now map those subjects back to integer positions in dataset.graphs
            subj_train_idx = np.array([i for i, subj in enumerate(dataset.subjects) if subj in train_subj])
            subj_test_idx = np.array([i for i, subj in enumerate(dataset.subjects) if subj in test_subj])
            
            # Build dataloaders
            dataloaders = build_dataloaders(
                dataset,
                train_idx=subj_train_idx,
                test_idx=subj_test_idx,
                gcn_mode=True,
                batch_size=params['batch_size'],
                node_level=node_level
            )

        else:
            dataloaders = build_dataloaders(
                dataset,
                train_idx,
                test_idx,
                gcn_mode=True,
                batch_size=params['batch_size'],
                node_level=node_level
            )
        train_loader = dataloaders["train"]
        test_loader = dataloaders["test"]

        # Setup device, model, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if node_level:
            model = NodeLevelGCN(in_channels=X.shape[1], hidden_dim=params["hidden_dim"]).to(device)
        else:
            model = GraphLevelGCN(hidden_dim=params["hidden_dim"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        # Loss function adjusted for class imbalance
        pos_weight = make_class_weight(y[train_idx], device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        if fold == 1 and learning_plot:
            outer_train_losses, test_losses = [], []

        # Training loop
        for epoch in range(params['epochs']):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                data.edge_attr = data.edge_attr.to(device)
                data.y = data.y.to(device)
                
                optimizer.zero_grad()
                if node_level:
                    out = model(data.x, data.edge_index, data.edge_attr)
                else:
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y.view(-1, 1).float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            epoch_loss = total_loss / len(train_loader)
            if fold == 1:
                print(f"[Fold {fold}] Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")
            if fold == 1 and learning_plot:
                outer_train_losses.append(epoch_loss)
                # Test losses
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for data in test_loader:
                        data = data.to(device)
                        if node_level:
                            out = model(data.x, data.edge_index, data.edge_attr)
                        else:
                            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                        loss = criterion(out, data.y.view(-1, 1).float())
                        test_loss += loss.item()
                test_losses.append(test_loss / len(test_loader))

        acc, auc = evaluate_gcn(model, test_loader, device, node_level=node_level)

        if fold == 1 and learning_plot:
            return acc, auc, outer_train_losses, test_losses
        return acc, auc
    

    ########################## 5. Cross Validation ##############################
    best_fold_one_params = cross_validation_control(
        X, 
        y, 
        subj_label, 
        train_config,
        train_one_fold, 
        test_one_fold, 
        model_name="GCN", 
        dataset_name=dataset_name,
        date_str=date_str,
        seed=seed
        )
    
    
    ########################## 6. Optional Retrain on Full Dataset #########################
    def full_retrain(dataset, best_params):
        full_loader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if node_level:
            final_model = NodeLevelGCN(in_channels=X.shape[1], hidden_dim=best_params["hidden_dim"], dropout=best_params["dropout"]).to(device)
        else:
            final_model = GraphLevelGCN(hidden_dim=best_params["hidden_dim"], dropout=best_params["dropout"]).to(device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        pos_weight = make_class_weight(y, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(best_params['epochs']):
            final_model.train()
            total_loss = 0
            for data in full_loader:
                data = data.to(device)
                data.edge_attr = data.edge_attr.to(device)
                data.y = data.y.to(device)
                optimizer.zero_grad()
                if node_level:
                    out = final_model(data.x, data.edge_index, data.edge_attr)
                else:
                    out = final_model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y.view(-1, 1).float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Full Dataset] Epoch {epoch+1} Train Loss: {total_loss/len(full_loader):.4f}")
        return final_model


    if train_config.get('retrain_full_dataset', False):
        print("Retraining on full dataset...")
        final_model = full_retrain(dataset, best_fold_one_params)
    
        final_model_path = os.path.join(model_config['output_dir'], f"model-{dataset_name}.pth")
        save_model(final_model, final_model_path)