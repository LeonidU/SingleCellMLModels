import os
import random
import numpy as np
from scipy.io import mmread
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def filter_underrepresented_samples(dataset, threshold=0.02):
    # Step 1: Count the occurrences of each label
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    num_labels = max(labels)
    label_counts = Counter(labels)

    # Step 2: Calculate the label frequencies
    total_samples = len(labels)
    label_frequencies = {label: count / total_samples for label, count in label_counts.items()}

    # Step 3: Identify labels that meet the frequency threshold
    valid_labels = {label for label, freq in label_frequencies.items() if freq >= threshold}
    filtered_labels_count = len(label_counts) - len(valid_labels)

    # Step 4: Filter out samples with labels below the threshold
    filtered_indices = [i for i in range(len(dataset)) if dataset[i][1].item() in valid_labels]

    # Step 5: Create a new dataset with filtered samples
    filtered_mtx = dataset.mtx[:, filtered_indices]
    filtered_col_names = [dataset.col_names[i] for i in filtered_indices]
    filtered_cells = dataset.cells[dataset.cells['column'].isin(filtered_col_names)]

    return MTXDataset(filtered_mtx, filtered_col_names, filtered_cells), filtered_mtx.shape[0], num_labels

class MTXDataset(Dataset):
    def __init__(self, mtx, col_names, cells):
        self.mtx = mtx
        self.col_names = col_names
        self.cells = cells

    def __len__(self):
        return self.mtx.shape[1]

    def __getitem__(self, idx):
        column_data = self.mtx[:, idx].flatten()
        column_name = self.col_names[idx]
        column_type = self.cells[self.cells['column'] == column_name]['type_encoded'].values[0]
        return torch.tensor(column_data, dtype=torch.float32), torch.tensor(column_type, dtype=torch.long)

# Custom Dataset class
def load_data(data_dir, features_path):
    all_mtx = []
    all_col_names = []
    all_cells = []
    common_features = None

    # Load features to filter rows
    with open(features_path, 'r') as f:
        features = set(line.strip().split(" ")[1].replace("\"", "") for line in f.readlines())

    label_encoder = LabelEncoder()
    combined_cell_types = []

    # Iterate through each subdirectory
    for sub_dir in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, sub_dir)
        if os.path.isdir(sub_path):
            # Automatically find files in the subdirectory
            mtx_path = None
            colnames_path = None
            cells_path = None
            rownames_path = None

            for file in os.listdir(sub_path):
                if file.endswith('.mtx'):
                    mtx_path = os.path.join(sub_path, file)
                elif file.endswith('.mtx_cols'):
                    colnames_path = os.path.join(sub_path, file)
                elif file.endswith('.cell_metadata.tsv'):
                    cells_path = os.path.join(sub_path, file)
                elif file.endswith('.mtx_rows'):
                    rownames_path = os.path.join(sub_path, file)

            if not (mtx_path and colnames_path and cells_path and rownames_path):
                continue

            # Load the MTX matrix
            mtx = mmread(mtx_path).tocsc()

            # Load column names
            with open(colnames_path, 'r') as f:
                col_names = [line.strip() for line in f.readlines()]

            # Load cell type mapping (use only the last column)
            cells = pd.read_csv(cells_path, sep='\t', header=0)
            cells = cells[['id', 'inferred_cell_type_-_ontology_labels']]
            cells.columns = ['column', 'type']
            combined_cell_types.extend(cells['type'].tolist())

            # Load row names (use only the first column)
            with open(rownames_path, 'r') as f:
                row_names = [line.strip().split('\t')[0] for line in f.readlines()]

            # Filter rows based on features
            valid_row_indices = [i for i, row in enumerate(row_names) if row in features]
            filtered_features = set(row_names[i] for i in valid_row_indices)

            if common_features is None:
                common_features = filtered_features
            else:
                common_features &= filtered_features

            all_mtx.append((mtx, valid_row_indices, row_names))
            all_col_names.extend(col_names)
            all_cells.append(cells)

    # Filter all matrices to keep only common features
    final_mtx_list = []
    for mtx, valid_row_indices, row_names in all_mtx:
        common_row_indices = [i for i in valid_row_indices if row_names[i] in common_features]
        filtered_mtx = mtx[common_row_indices, :]
        final_mtx_list.append(filtered_mtx.toarray())

    # Stack all matrices horizontally
    combined_mtx = np.hstack(final_mtx_list)
    combined_cells = pd.concat(all_cells, ignore_index=True)

    # Encode cell types
    combined_cells['type_encoded'] = label_encoder.fit_transform(combined_cell_types)
    dataset = MTXDataset(torch.tensor(combined_mtx, dtype=torch.float32), all_col_names, combined_cells)
    dataset, input_features, input_classes = filter_underrepresented_samples(dataset)
    print("input features", input_features)
    print("input_classes", input_classes)
    return input_features, input_classes, dataset

#if __name__ == "__main__":
#    data_dir = "learning_set"
#    features_path = "Hsapiens_features.txt"
#    
#    dataset = load_data(data_dir, features_path)
#    
#    # Create DataLoader
#    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#    
#    # Sample one batch from DataLoader
#    for X, y in dataloader:
#        print("Sampled Column Data (X):", X.flatten().numpy())
#        print("Column Type (y):", y.item())
#        break

