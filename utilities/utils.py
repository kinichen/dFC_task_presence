import math
import numpy as np
import pandas as pd
import os
import torch



def get_n_ROI(a, b, c): # solves quadratic in ax**2+bx+c=0 form.
	discriminant = b**2 - 4*a*c
	if discriminant < 0:
		return ValueError("No real roots")
	root1 = (-b + math.sqrt(discriminant)) / (2*a)  # always returns a float
	root2 = (-b - math.sqrt(discriminant)) / (2*a)
	
	if root1 > 0:
		if root1.is_integer():
			return int(root1)
		else:
			return ValueError(f"Number of ROIs = {root1} is not an integer")
	else:
		if root2.is_integer():
			return int(root2)
		else:
			return ValueError(f"Number of ROIs = {root2} is not an integer")


def vec_to_symmetric_matrix(vec, roi):
	mat = np.zeros((roi, roi))
	idx = np.triu_indices(roi, k=1)	# excludes diagonal
	mat[idx] = vec
	mat = mat + mat.T
	return mat


def harmonize_TR(datasets: list):
    ''' 
    Harmonize (truncate) samples across different datasets using their TR labels. 
    Same data assessed by different dFC methods gives different number of time points.
    
    Args: 
    	datasets (list): a list of dFC dictionaries, each loaded from config.yaml. 
    
    Returns: 
    	harmonized (list): a list of dFC dictionaries with samples that are not 
    	in all three datasets removed. 
    '''
    dfs = []
    for ds in datasets:
        df = pd.DataFrame({
            "subj": ds["subj_label"],
            "TR": ds["TR_labels"],
            "y": ds["y"],
            "X": list(ds["X"])  # keep as object so rows align easily
        })
        # Critical to use (subj, TR) pair as index since different subjects can have
        # different number of time points (TRs), shifting the labels, which causes
        # a mismatch of samples across datasets
        df.set_index(["subj", "TR"], inplace=True)
        dfs.append(df)

    # Find intersection of (subj, TR) pairs
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    harmonized = []
    for df in dfs:
        df = df.loc[common_idx]	# label-based indexing
        harmonized.append({
            "subj_label": df.index.get_level_values("subj").to_numpy(),
            "TR_labels": df.index.get_level_values("TR").to_numpy(),
            "y": df["y"].to_numpy(),
            "X": np.stack(df["X"].to_numpy())  # back to ndarray
        })

    # Check
    for i in range(1, len(harmonized)):
        assert np.array_equal(harmonized[0]["y"], harmonized[i]["y"]), \
            "Mismatch in y labels after harmonization. Bad!!"

    return harmonized


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Final model saved to {path}")