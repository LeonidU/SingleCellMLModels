import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import dataset
import numpy as np
from torcheval.metrics import MulticlassAUROC
import torch.nn.functional as F

# Define the 1D CNN model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
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
        if self.shortcut:
            x = self.shortcut(x)
        out += x
        out = F.relu(out)
        return out

class SingleCellResNet(nn.Module):
    def __init__(self, input_features, num_classes):
        super(SingleCellResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, 1024, num_blocks=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
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


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def evaluate_model(model, test_loader, size):
    model.eval()
    all_labels = []
    all_outputs = []
    correct, n = 0.0, 0
    with torch.no_grad():
        for vectors, labels in test_loader:
            vectors, labels = vectors.to(device), labels.to(device)
            outputs = model(vectors)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(np.array(np.argmax(outputs.cpu().numpy(), axis=1)))
            predicted = np.array(np.argmax(outputs.cpu().numpy(), axis=1))
            correct += (predicted == labels.cpu().numpy()).sum().item()
            n += labels.size(0)
    all_outputs_np = np.array(all_outputs)
    all_labels_np = np.array(all_labels)
    f1 = f1_score(all_labels_np, all_outputs_np, average='weighted')
    accuracy = accuracy_score(all_labels_np, all_outputs_np)
    recall = recall_score(all_labels_np, all_outputs_np, average='weighted')
    precision = precision_score(all_labels_np, all_outputs_np, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    auc = 0.0
    return auc, accuracy, precision, recall, f1

def train_model(model, train_loader, test_loader, criterion, optimizer, input_classes, num_epochs=20, patience=5):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            optimizer.zero_grad()
            assert not torch.isnan(inputs).any(), "Inputs contain NaN values"
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
        auc, accuracy, precision, recall, f1 = evaluate_model(model, test_loader, input_classes)
    torch.save(model.state_dict(), "convnet1d_model.pth")

features_path = "Hsapiens_features.txt"
dir = "../learning_set/liver/"
input_length, input_classes, dataset, weights = dataset.load_data(dir, features_path)
print("input_classes:", input_classes)
print("input_length:", input_length)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SingleCellResNet(input_features=input_length, num_classes=input_classes)
#model.apply(weights_init)
model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
#FocalLoss(alpha=1.9, gamma=2.6)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train_loader, test_loader = split_dataset(dataset)
train_loader, test_loader = DataLoader(train_loader, batch_size=16, shuffle=True), DataLoader(test_loader, batch_size=16, shuffle=True)
train_model(model, train_loader, test_loader, criterion, optimizer, input_classes, num_epochs=200)

