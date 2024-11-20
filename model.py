import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader, random_split
#from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import dataset
import numpy as np

import torch.nn.functional as F
# Define the 1D CNN model

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SingleCellResNet(nn.Module):
    def __init__(self, input_features, num_classes):
        super(SingleCellResNet, self).__init__()
        super(SingleCellResNet, self).__init__()
        # Embedding layer to reduce dimensionality
        self.embedding_layer = nn.Linear(input_features, 512)
        self.input_layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding_layer(x)  # Map to lower-dimensional space
        x = x.unsqueeze(1)  # Add channel dimension if necessary
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Initialize the model, loss function, and optimizer
if torch.cuda.is_available():
    print("CUDA GPU is available.")
else:
    print("CUDA GPU is not available.")

#mtx_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx"
#colnames_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx_cols"
#cells_path = "../E-ANND-2/E-ANND-2.cells.txt"
#rownames_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx_rows"
features_path = "Hsapiens_features.txt"
dir = "../learning_set/"
input_length, input_classes, dataset = dataset.load_data(dir, features_path)
# input_features, input_classes, dataset

print(input_classes)
print(input_length)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SingleCellResNet(input_features=input_length,  num_classes=input_classes)
model.apply(weights_init)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate_model(model, test_loader, size):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for vectors, labels in test_loader:
#            print(vectors)
            vectors, labels = vectors.to(device), labels.to(device)

            # Forward pass
            outputs = model(vectors)
#            print(outputs)
#            print(labels)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    # Convert outputs to predicted labels
    all_outputs_np = np.array(all_outputs)
    all_labels_np = np.array(all_labels)
    predicted_labels = np.argmax(all_outputs_np, axis=1)
    np.savetxt("all_outputs_np.csv", all_outputs_np, delimiter=",")
    np.savetxt("all_labels_np.csv", all_labels_np, delimiter=",")
    np.savetxt("predicted_labels.csv", predicted_labels, delimiter=",")

    # Calculate metrics
#    unique_classes = np.unique(all_labels_np)
#    if len(unique_classes) < all_outputs_np.shape[1]:
#        # Pad the output with zeros for missing classes
#        padded_outputs = np.zeros((all_outputs_np.shape[0], len(unique_classes)))
#        for i, cls in enumerate(unique_classes):
#            padded_outputs[:, i] = all_outputs_np[:, cls]
#        all_outputs_np = padded_outputs
#    auc = roc_auc_score(all_labels_np, all_outputs_np[:, unique_classes], multi_class='ovr', labels=unique_classes)
    auc = roc_auc_score(all_labels_np, all_outputs_np, multi_class='ovr')
#    auc = roc_auc_score(all_labels_np, all_outputs_np, multi_class='ovr', labels=unique_classes)
    accuracy = accuracy_score(all_labels_np, predicted_labels)
    precision = precision_score(all_labels_np, predicted_labels, average='weighted')
    recall = recall_score(all_labels_np, predicted_labels, average='weighted')
    f1 = f1_score(all_labels_np, predicted_labels, average='weighted')

    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")


# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, input_classes, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_labels = []
        all_outputs = []
        for inputs, labels in train_loader:
            # Move data to the device (GPU or CPU)
#            inputs, labels = inputs.float(), labels
#            inputs, labels = inputs.permute(0, 2, 1).float().to(device), labels.long().to(device)
            inputs, labels = inputs.float().to(device), labels.to(device)
            # Zero the gradient
            optimizer.zero_grad()
#            print(labels.shape)
            # Forward pass
            outputs = model(inputs)
#            print(outputs.shape)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            all_labels.extend(labels.cpu().numpy())
#            all_outputs.extend(outputs.cpu().detach().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().detach().numpy())
#            print(F.softmax(outputs, dim=1).cpu().detach().numpy())
#            print(inputs)
#            auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr')
#            print(f"AUC is {auc:.4f}")
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
#        all_outputs_np = np.array(all_outputs)
#        all_labels_np = np.array(all_labels)
#        auc = roc_auc_score(all_labels_np, all_outputs_np, multi_class='ovr')
#        print(f'AUC {auc}')
        evaluate_model(model, test_loader, input_classes)
        
    torch.save(model.state_dict(), "convnet1d_model.pth")

train_loader, test_loader = split_dataset(dataset)
train_loader, test_loader = DataLoader(train_loader, batch_size=32, shuffle=True), DataLoader(test_loader, batch_size=32, shuffle=True)
train_model(model, train_loader, test_loader, criterion, optimizer, input_classes, num_epochs=10)
#evaluate_model(model, test_loader)
