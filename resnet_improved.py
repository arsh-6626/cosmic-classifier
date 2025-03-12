import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split

# Define ResNet-inspired MLP with Skip Connections
class ResNetMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.3):
        super(ResNetMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x1 = self.relu(self.batch_norm1(self.fc1(x)))
        x1 = self.dropout(x1)
        
        x2 = self.relu(self.batch_norm2(self.fc2(x1)))
        x2 = self.dropout(x2)
        
        # Skip Connection
        x2 += x1
        
        out = self.fc3(x2)
        return out

# Generate Dummy Dataset
X = np.random.rand(5000, 20).astype(np.float32)  # 5000 samples, 20 features each
y = np.random.randint(0, 2, size=(5000, 1)).astype(np.float32)  # Binary classification

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
tensor_x_train = torch.tensor(X_train)
tensor_y_train = torch.tensor(y_train)
tensor_x_val = torch.tensor(X_val)
tensor_y_val = torch.tensor(y_val)

# Dataloader
dataset_train = TensorDataset(tensor_x_train, tensor_y_train)
dataset_val = TensorDataset(tensor_x_val, tensor_y_val)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

# Model Initialization
input_size = 20
hidden_size = 64
output_size = 1
model = ResNetMLP(input_size, hidden_size, output_size)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 Regularization

# Early Stopping Setup
def early_stopping(val_losses, patience=5):
    if len(val_losses) > patience and val_losses[-1] > min(val_losses[-patience:]):
        return True
    return False

# Training Loop
epochs = 100
train_losses, val_losses = [], []
patience = 10  # Early stopping patience

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(dataloader_train)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader_val:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs.unsqueeze(1), labels)

            val_loss += loss.item()
    val_loss /= len(dataloader_val)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Check early stopping condition
    if early_stopping(val_losses, patience):
        print("Early stopping triggered!")
        break
