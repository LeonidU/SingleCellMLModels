import random
import numpy as np
from scipy.io import mmread
import pandas as pd

# Load MTX file, columns, rows, and cell mapping
def load_data(mtx_path, colnames_path, cells_path, rownames_path, features_path):
    # Load the MTX matrix
    mtx = mmread(mtx_path).tocsc()
    
    # Load column names
    with open(colnames_path, 'r') as f:
        col_names = [line.strip() for line in f.readlines()]
    
    # Load cell type mapping (use only the last column)
    cells = pd.read_csv(cells_path, sep='\t', header=0, usecols=[-1], names=['type'])
    cells['column'] = pd.read_csv(cells_path, sep='\t', header=0, usecols=[0], names=['column'])
    
    # Load row names (use only the first column)
    with open(rownames_path, 'r') as f:
        row_names = [line.strip().split('\t')[0] for line in f.readlines()]
    
    # Load features to filter rows
    with open(features_path, 'r') as f:
        features = set(line.strip() for line in f.readlines())
    
    # Filter rows based on features
    valid_row_indices = [i for i, row in enumerate(row_names) if row in features]
    mtx = mtx[valid_row_indices, :]
    
    return mtx, col_names, cells

def sample_column(mtx, col_names, cells):
    # Randomly select one column index
    col_index = random.randint(0, mtx.shape[1] - 1)
    
    # Extract the column data
    column_data = mtx[:, col_index].toarray().flatten()
    
    # Get the column name and its type
    column_name = col_names[col_index]
    column_type = cells[cells['column'] == column_name]['type'].values[0]
    
    return column_data, column_type

def data_sampling(mtx_path, colnames_path, cells_path, rownames_path, features_path="Hsapiens_features.txt")
     mtx, col_names, cells = load_data(mtx_path, colnames_path, cells_path, rownames_path, features_path)
    column_data, column_type = sample_column(mtx, col_names, cells)
    
    print("Sampled Column Data:", column_data)
    print("Column Type:", column_type)

data_sampling("E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx", "E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx_cols", "E-ANND-2/E-ANND-2.cells.txt", "E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx_rows")
