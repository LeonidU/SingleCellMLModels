import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import dataset
import numpy as np
from sklearn.cluster import KMeans
from torcheval.metrics import MulticlassAUROC
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Define Temporal Convolutional Network (TCN) block
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        if res.size(2) != out.size(2):
            res = F.interpolate(res, size=out.size(2))  # Adjust residual size if mismatch occurs
        return self.relu(out + res)

# Define TCN model
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]  # Fix input channels to match data input
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], n_clusters)

    def forward(self, x):
        y1 = self.network(x)
        y1 = y1.mean(dim=2)  # Global average pooling
        return self.linear(y1)


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

# Initialize the model, loss function, and optimizer
if torch.cuda.is_available():
    print("CUDA GPU is available.")
else:
    print("CUDA GPU is not available.")

features_path = "Hsapiens_features.txt"
dir = "../test_ls/"
input_dim, n_clusters, dataset = dataset.load_data(dir, features_path)
train_loader, test_loader = split_dataset(dataset)
train_loader, test_loader = DataLoader(train_loader, batch_size=32, shuffle=True), DataLoader(test_loader, batch_size=32, shuffle=True)

# Model parameters
hidden_dim = 64
num_channels = [hidden_dim] * 3  # Three layers with the same number of channels
kernel_size = 3
dropout = 0.2

# Create TCN model
model = TCN(num_inputs=input_dim, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        data = data.unsqueeze(1)  # Add channel dimension
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            data = data.unsqueeze(1)  # Add channel dimension
            outputs = model(data)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    all_outputs_np = np.array(all_outputs)
    all_labels_np = np.array(all_labels)
    predicted_labels = np.argmax(all_outputs_np, axis=1)

    # Calculate metrics
    auc = roc_auc_score(all_labels_np, all_outputs_np, multi_class='ovr')
    accuracy = accuracy_score(all_labels_np, predicted_labels)
    precision = precision_score(all_labels_np, predicted_labels, average='weighted')
    recall = recall_score(all_labels_np, predicted_labels, average='weighted')
    f1 = f1_score(all_labels_np, predicted_labels, average='weighted')

    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    return auc, accuracy, precision, recall, f1

evaluate_model(model, test_loader)

