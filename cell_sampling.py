import random
import numpy as np
from scipy.io import mmread
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Custom Dataset class
def load_data(mtx_path, colnames_path, cells_path, rownames_path, features_path):
    class MTXDataset(Dataset):
        def __init__(self, mtx, col_names, cells):
            self.mtx = mtx
            self.col_names = col_names
            self.cells = cells

        def __len__(self):
            return self.mtx.shape[1]

        def __getitem__(self, idx):
            column_data = self.mtx[:, idx].toarray().flatten()
            column_name = self.col_names[idx]
            column_type = self.cells[self.cells['column'] == column_name]['type_encoded'].values[0]
            return torch.tensor(column_data, dtype=torch.float32), torch.tensor(column_type, dtype=torch.long)

    # Load the MTX matrix
    mtx = mmread(mtx_path).tocsc()
    print(mtx)
    # Load column names
    with open(colnames_path, 'r') as f:
        col_names = [line.strip() for line in f.readlines()]
    
    # Load cell type mapping (use only the last column)
    cells = pd.read_csv(cells_path, sep='\t', header=0)
    cells = cells[['Cell ID', 'inferred cell type - ontology labels']]
    cells.columns = ['column', 'type']
    
    # Encode cell types
    label_encoder = LabelEncoder()
    cells['type_encoded'] = label_encoder.fit_transform(cells['type'])
    
    # Load row names (use only the first column)
    with open(rownames_path, 'r') as f:
        row_names = [line.strip().split('\t')[0] for line in f.readlines()]
    
    # Load features to filter rows
    with open(features_path, 'r') as f:
        features = set(line.strip().split(" ")[1].replace("\"", "") for line in f.readlines())

    # Filter rows based on features
    valid_row_indices = [i for i, row in enumerate(row_names) if row in features]
    mtx = mtx[valid_row_indices, :]
    
    dataset = MTXDataset(mtx, col_names, cells[['column', 'type_encoded']])
    return dataset

if __name__ == "__main__":
    mtx_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx"
    colnames_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx_cols"
    cells_path = "../E-ANND-2/E-ANND-2.cells.txt"
    rownames_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx_rows"
    features_path = "Hsapiens_features.txt"
    
    dataset = load_data(mtx_path, colnames_path, cells_path, rownames_path, features_path)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Sample one batch from DataLoader
    for X, y in dataloader:
        print("Sampled Column Data (X):", X.flatten().numpy())
        print("Column Type (y):", y.item())
        break

