import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import dataset
import numpy as np
from sklearn.cluster import KMeans

# Define the Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

    def forward(self, x):
        # Squeeze
        y = torch.mean(x, dim=-1)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.unsqueeze(-1)
        return x * y

# Define the 1D CNN model with SE blocks
class SE1DCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_clusters):
        super(SE1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.se1 = SEBlock(hidden_dim)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.se2 = SEBlock(hidden_dim // 2)
        self.fc = nn.Linear((hidden_dim // 2) * input_dim, n_clusters)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.se1(x)
        x = F.relu(self.conv2(x))
        x = self.se2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        focal_weight = (1 - probs) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(probs)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Initialize the model, loss function, and optimizer
if torch.cuda.is_available():
    print("CUDA GPU is available.")
else:
    print("CUDA GPU is not available.")

features_path = "Hsapiens_features.txt"
#dir = "../test_ls/"
dir = "../learning_set/liver/"
input_dim, n_clusters, dataset, weights = dataset.load_data(dir, features_path)
train_loader, test_loader = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_loader, test_loader = DataLoader(train_loader, batch_size=16, shuffle=True), DataLoader(test_loader, batch_size=16, shuffle=True)

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    all_outputs_np = np.array(all_outputs)
    all_labels_np = np.array(all_labels)
    predicted_labels = np.argmax(all_outputs_np, axis=1)
    auc = 0.0
#    auc = roc_auc_score(all_labels_np, all_outputs_np, multi_class='ovr')
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

hidden_dim = 64
model = SE1DCNN(input_dim, hidden_dim, n_clusters)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize optimizer and loss function
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
epochs = 200
max = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
    auc, acc, precision, recall, f1 = evaluate_model(model, test_loader)
    if (acc > max):
        max = acc
        torch.save(model.state_dict(), f"se.pth")
        print("model saved")

# Evaluation function

evaluate_model(model, test_loader)

