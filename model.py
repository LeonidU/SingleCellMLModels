import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cell_sampling
# Define the 1D CNN model
class CNN1DClassifier(nn.Module):
    def __init__(self, input_length, num_classes=21):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Calculate the output length after convolutions and pooling
        conv_output_length = input_length // 8
        self.fc1 = nn.Linear(64 * conv_output_length, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Initialize the model, loss function, and optimizer

mtx_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx"
colnames_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx_cols"
cells_path = "../E-ANND-2/E-ANND-2.cells.txt"
rownames_path = "../E-ANND-2/E-ANND-2.aggregated_filtered_normalised_counts.mtx_rows"
features_path = "Hsapiens_features.txt"
num_classes, input_length, dataset = cell_sampling.load_data(mtx_path, colnames_path, cells_path, rownames_path, features_path)
print(num_classes)
print(input_length)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = CNN1DClassifier(input_length=input_length,  num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # Move data to the device (GPU or CPU)
            inputs, labels = inputs.float(), labels.long()

            # Zero the gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update loss
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')



train_model(model, dataloader, criterion, optimizer, num_epochs=100)
