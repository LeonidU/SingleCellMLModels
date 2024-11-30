import torch
import torch.nn as nn
import torch.optim as optim
import dataset
from torch.utils.data import DataLoader, random_split

class WaveNet(nn.Module):
    def __init__(self, input_length, n_classes, n_blocks=2, n_layers=4, residual_channels=8, skip_channels=8, kernel_size=2):
        super(WaveNet, self).__init__()
        self.residual_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        for _ in range(n_blocks):
            for i in range(n_layers):
                dilation = 2 ** i
                self.residual_layers.append(nn.Conv1d(in_channels=residual_channels, out_channels=residual_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation))
                self.skip_layers.append(nn.Conv1d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=1))

        self.start_conv = nn.Conv1d(in_channels=1, out_channels=residual_channels, kernel_size=1)
        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels, out_channels=skip_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(in_channels=skip_channels, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (B, C, T)
        x = self.start_conv(x)

        skip_connections = []

        for res_layer, skip_layer in zip(self.residual_layers, self.skip_layers):
            residual = res_layer(x)
            gated = torch.tanh(residual) * torch.sigmoid(residual)
            # Adjust the size of gated to match x if necessary
            if gated.size(-1) != x.size(-1):
                gated = gated[:, :, :x.size(-1)]
            skip = skip_layer(gated)
            skip_connections.append(skip)
            x = x + gated  # Residual connection

        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.mean(dim=-1)  # Average over the temporal dimension for classification
        return x


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

# Initialize the model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features_path = "Hsapiens_features.txt"
#dir = "../learning_set/lung/"
dir = "../test_ls/"
input_length, input_classes, dataset = dataset.load_data(dir, features_path)
train_loader, test_loader = split_dataset(dataset)
n_classes = input_classes
train_loader, test_loader = DataLoader(train_loader, batch_size=32, shuffle=True), DataLoader(test_loader, batch_size=32, shuffle=True)

# input_features, input_classes, dataset

print(f"input classes : {input_classes}")
print(f"input length : {input_length}")

model = WaveNet(input_length=input_length, n_classes=n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 500
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

