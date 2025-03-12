import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE

# Load dataset
train_file = "cosmic-classifier-cogni25/data/cosmicclassifierTraining.csv"
data = pd.read_csv(train_file)

expected_columns = ["Atmospheric_Density", "Surface_Temperature", "Gravity",
                    "Water_Content", "Mineral_Abundance", "Orbital_Period",
                    "Proximity_to_Star", "Magnetic_Field_Strength", "Radiation_Levels",
                    "Atmospheric_Composition_Index", "Planet_Class"]

if list(data.columns) != expected_columns:
    data.columns = expected_columns

# Data Cleaning
data = data.dropna()
data_clean = data[data["Planet_Class"].apply(lambda x: isinstance(x, (int, float)) and x >= 0)]

# Feature and Label Separation
features = data_clean.drop("Planet_Class", axis=1)
non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns

for col in non_numeric_cols:
    features[col] = features[col].str.extract('(\d+)').astype(float)

X = features.values
y = data_clean["Planet_Class"]

# Encoding labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scaling Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Dataloader
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ResNet-MLP Model
class ResNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResNetMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Residual Block 1
        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn2 = nn.BatchNorm1d(hidden_dim)

        # Residual Block 2
        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_bn1 = nn.BatchNorm1d(hidden_dim)
        self.res2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_bn2 = nn.BatchNorm1d(hidden_dim)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))

        # Residual Block 1
        identity = x
        out = torch.relu(self.res1_bn1(self.res1_fc1(x)))
        out = self.res1_bn2(self.res1_fc2(out))
        x = torch.relu(out + identity)  # Skip Connection

        # Residual Block 2
        identity = x
        out = torch.relu(self.res2_bn1(self.res2_fc1(x)))
        out = self.res2_bn2(self.res2_fc2(out))
        x = torch.relu(out + identity)  # Skip Connection

        # Final output
        x = self.fc_out(x)
        return x

# Initialize Model
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = len(le.classes_)

model = ResNetMLP(input_dim, hidden_dim, output_dim)
print(model)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 75
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    val_acc = accuracy_score(y_val, all_preds)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Plot Losses
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

final_acc = accuracy_score(y_val, preds)
print("Final Validation Accuracy: {:.2f}%".format(final_acc * 100))

# Classification Report
target_names = [str(x) for x in le.classes_]
print("\nClassification Report:")
print(classification_report(y_val, preds, target_names=target_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, preds)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# t-SNE Visualization
with torch.no_grad():
    embeddings = model.fc1(X_val_tensor)
    embeddings = embeddings.cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(8,6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_val, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title("t-SNE of Validation Embeddings (True Labels)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

# Save Model
torch.save(model.state_dict(), "cosmic_classifier_resnet_mlp.pth")
print("Model and preprocessing objects saved.")
