import dataset
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

# Autoencoder Definition
class Autoencoder(nn.Module):
    def __init__(self, input_length, encoded_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_length, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, encoded_dim),
            nn.ReLU()

        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, input_length),
#            nn.Softmax()
#            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Classifier Definition
class Classifier(nn.Module):
    def __init__(self, encoded_dim=128, num_classes=6):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
#            nn.Softmax()
        )

    def forward(self, x):
        return self.classifier(x)

def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Evaluate Classifier
def evaluate_model(classifier, autoencoder, data_loader):
    classifier.eval() 
    autoencoder.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device).float()
#            print(data.shape)
            labels = labels.to(device).long()  
            encoder = autoencoder.encoder(data)
#            print(encoder.shape)
            outputs = classifier(encoded)
#            print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
#            print(outputs)
#            correct += (predicted == labels).sum().item()
#            if predicted.size(0) == labels.size(0):  # Ensure batch sizes match
            correct += (predicted == labels).sum().item()
#            print(predicted == labels)
            total += labels.size(0)
    return 100 * correct / total


# Initialize the model, loss function, and optimizer
if torch.cuda.is_available():
    print("CUDA GPU is available.")
else:
    print("CUDA GPU is not available.")

features_path = "Hsapiens_features.txt"
#dir = "../learning_set/lung/"
dir = "../test_ls/"
input_length, input_classes, dataset = dataset.load_data(dir, features_path)
train_loader, test_loader = split_dataset(dataset)
#train_loader, test_loader = DataLoader(train_loader, batch_size=16, shuffle=True), DataLoader(test_loader, batch_size=16, shuffle=True)
train_loader = DataLoader(train_loader, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_loader, batch_size=16, shuffle=True, drop_last=True)


# input_features, input_classes, dataset

print(f"input classes : {input_classes}")
print(f"input length : {input_length}")

encoded_dim = 128  # Dimension of the encoded representation
num_epochs_autoencoder = 20
num_epochs_classifier = 50
learning_rate = 1e-3



# Instantiate Autoencoder and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder(input_length, encoded_dim).to(device)
criterion_reconstruction = nn.MSELoss()
criterion_sparsity = lambda encoded: torch.mean(torch.abs(encoded))  # L1 penalty for sparsity
lambda_sparsity = 1e-3  # Weight for sparsity loss
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Train Autoencoder
print("Training Autoencoder...")
for epoch in range(num_epochs_autoencoder):
    autoencoder.train()
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device).float()
        encoded, decoded = autoencoder(data)
        reconstruction_loss = criterion_reconstruction(decoded, data)
#        print(decoded)
#        print(data)
#        sparsity_loss = lambda_sparsity * criterion_sparsity(encoded)
        loss = reconstruction_loss # + sparsity_loss

        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs_autoencoder}], Loss: {total_loss / len(train_loader):.4f}")

# Freeze Encoder for Feature Extraction
autoencoder.eval()
for param in autoencoder.encoder.parameters():
    param.requires_grad = False

# Instantiate Classifier and Optimizer
classifier = Classifier(encoded_dim, input_classes).to(device)
criterion_classifier = nn.CrossEntropyLoss()
clf_optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=learning_rate)
#criterion_classifier = nn.MSELoss() #ClassifierLoss()
print("Training Classifier...")
best_accuracy = 0
patience_counter = 0
patience = 40
for epoch in range(num_epochs_classifier):
    classifier.train()
    total_loss = 0
    for data, labels in train_loader:
        data = data.to(device).float()
        labels = labels.to(device).long()

        # Extract Features Using Encoder
        with torch.no_grad():
            encoded = autoencoder.encoder(data)

        # Classify Using the Extracted Features
        outputs = classifier(encoded)
#        print(torch.mean(outputs, 1))
#        _ , outputs = torch.max(outputs.data, 1)
        loss = criterion_classifier(outputs, labels)

        clf_optimizer.zero_grad()
        loss.backward()
        clf_optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs_classifier}], Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate on Test Set
    current_accuracy = evaluate_model(classifier, autoencoder, test_loader)
    print(f'Validation Accuracy: {current_accuracy:.2f}%')

    # Early Stopping Check
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        patience_counter = 0
        torch.save(classifier.state_dict(), 'best_classifier.pth')  # Save the best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model for final evaluation
classifier.load_state_dict(torch.load('best_classifier.pth'))


accuracy = evaluate_model(classifier, autoencoder, test_loader)
print(f'Classifier Accuracy on Test Set: {accuracy:.2f}%')

