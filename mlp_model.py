import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import joblib

# Class names mapping
class_names = {
    0: "Bewohnbar",        # Habitable
    1: "Terraformierbar",  # Terraformable
    2: "Rohstoffreich",    # Resource-rich
    3: "Wissenschaftlich", # Scientific
    4: "Gasriese",         # Gas giant
    5: "Wüstenplanet",     # Desert planet
    6: "Eiswelt",          # Ice world
    7: "Toxischetmosäre",  # Toxic atmosphere
    8: "Hohestrahlung",    # High radiation
    9: "Toterahswelt"      # Dead world
}

# Improved MLP model
# class EnhancedMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
#         super(EnhancedMLP, self).__init__()
        
#         layers = []
#         prev_dim = input_dim
        
#         # Create multiple hidden layers with different dimensions
#         for i, dim in enumerate(hidden_dims):
#             layers.append(nn.Linear(prev_dim, dim))
#             layers.append(nn.BatchNorm1d(dim))
            
#             # Use different activation functions in different layers
#             if i % 2 == 0:
#                 layers.append(nn.SiLU())
#             else:
#                 layers.append(nn.GELU())
                
#             layers.append(nn.Dropout(dropout_rate))
#             prev_dim = dim
        
#         # Output layer
#         layers.append(nn.Linear(prev_dim, output_dim))
        
#         self.model = nn.Sequential(*layers)
        
#         # Apply custom weight initialization
#         self._initialize_weights()
# Update the EnhancedMLP class:
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.4):
        super(EnhancedMLP, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        self.projection_layers = nn.ModuleList()  # Add projection layers for skip connections
        
        for i in range(len(hidden_dims)):
            if i == 0:
                # First block takes input from input_layer
                self.res_blocks.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i]),
                        nn.BatchNorm1d(hidden_dims[i]),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(hidden_dims[i], hidden_dims[i]),
                        nn.BatchNorm1d(hidden_dims[i]),
                    )
                )
                # No projection needed for the first block
                self.projection_layers.append(nn.Identity())
            else:
                # Subsequent blocks
                self.res_blocks.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                        nn.BatchNorm1d(hidden_dims[i]),
                        nn.GELU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(hidden_dims[i], hidden_dims[i]),
                        nn.BatchNorm1d(hidden_dims[i]),
                    )
                )
                # Add projection layer if dimensions change
                if hidden_dims[i] != hidden_dims[i-1]:
                    self.projection_layers.append(
                        nn.Sequential(
                            nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                            nn.BatchNorm1d(hidden_dims[i]),
                        )
                    )
                else:
                    self.projection_layers.append(nn.Identity())  # No projection needed
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        print("After input layer:", x.shape)
        
        for block, projection in zip(self.res_blocks, self.projection_layers):
            residual = projection(x)  # Project residual to match dimensions
            print("Residual shape:", residual.shape)
            x = block(x)
            print("After block:", x.shape)
            x += residual  # Skip connection
            x = F.gelu(x)
        
        print("Before output layer:", x.shape)
        return self.output_layer(x)
         
        return self.output_layer(x)        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    # def forward(self, x):
    #     return self.model(x)

def train_model(preprocessed_file):
    """
    Train MLP model on preprocessed data
    
    Args:
        preprocessed_file: Path to the preprocessed CSV file
    """
    print(f"Loading preprocessed data from {preprocessed_file}...")
    data = pd.read_csv(preprocessed_file)
    
    # Split features and labels
    X = data.drop('Planet_Class', axis=1).values
    y = data['Planet_Class'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Encoded classes:", le.classes_)
    
    # Save feature column names for later use
    feature_names = data.drop('Planet_Class', axis=1).columns.tolist()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    
    # Create more balanced datasets with focused augmentation for confused classes
    # print("\nPerforming targeted data augmentation for frequently confused classes...")
    
    # # Identify samples from most confused classes (4 & 9)
    # class4_indices = np.where(y_train == 4)[0]
    # class9_indices = np.where(y_train == 9)[0]

    # # Create synthetic samples with small perturbations for these classes
    # aug_samples = []
    # aug_labels = []

    # for idx in np.concatenate([class4_indices, class9_indices]):
    #     for _ in range(2):  # Create 2 synthetic samples per original
    #         noise = np.random.normal(0, 0.05, X_train.shape[1])  # Small perturbations
    #         perturbed = X_train[idx] + noise
    #         aug_samples.append(perturbed)
    #         aug_labels.append(y_train[idx])
    
    # Replace the augmentation section with:
    print("\nPerforming advanced data augmentation...")

    # Identify confused classes from confusion matrix (4,9,3,7,8)
    confused_classes = [3, 4, 7, 8, 9]
    aug_samples = []
    aug_labels = []

    for class_idx in confused_classes:
        class_indices = np.where(y_train == class_idx)[0]
        if len(class_indices) == 0:
            continue
        
        # Generate more samples for minority classes
        for idx in class_indices:
            for _ in range(3 if class_idx in [4,9] else 2):
                # Gaussian noise with class-specific variance
                noise_level = 0.1 if class_idx in [4,9] else 0.07
                noise = np.random.normal(0, noise_level, X_train.shape[1])
                
                # Feature-specific augmentation for key attributes
                if class_idx == 4:  # Gasriese
                    noise[0] *= 2  # Atmospheric_Density
                    noise[6] *= 2  # Proximity_to_Star
                elif class_idx == 9:  # Toterahswelt
                    noise[8] *= 3  # Radiation_Levels
                
                aug_samples.append(X_train[idx] + noise)
                aug_labels.append(class_idx)

    # Add mixup augmentation
    for _ in range(len(X_train)//2):
        idx1, idx2 = np.random.choice(len(X_train), 2, replace=False)
        alpha = 0.4
        mixed_sample = alpha*X_train[idx1] + (1-alpha)*X_train[idx2]
        aug_samples.append(mixed_sample)
        aug_labels.append(y_train[idx1] if np.random.rand() < alpha else y_train[idx2])

    # Add augmented samples to training data
    X_train_aug = np.vstack([X_train, np.array(aug_samples)])
    y_train_aug = np.concatenate([y_train, np.array(aug_labels)])
    print(f"Training data shape after augmentation: {X_train_aug.shape}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_aug, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

    # Define model parameters
    input_dim = X_train_aug.shape[1]
    hidden_dims = [256,128,96,64,32]  # Deeper network with varying layer sizes
    output_dim = len(le.classes_)
    dropout_rate = 0.25  # Custom dropout rate

    # Create the model
    model = EnhancedMLP(input_dim, hidden_dims, output_dim, dropout_rate)
    print(model)

    # Calculate class weights for weighted loss function
    class_samples = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = torch.tensor(
        [total_samples / (len(class_samples) * count) for count in class_samples],
        dtype=torch.float32
    )
    print("Class weights:", class_weights)

    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Use AdamW optimizer with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training parameters
    num_epochs = 100
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    print("\nTraining the model...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        val_acc = accuracy_score(all_true, all_preds)
        val_accuracies.append(val_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(epoch_val_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Detailed metrics for best model
            print("\nIntermediate Classification Report:")
            target_names = [f"{i}: {class_names[i]}" for i in range(len(class_names))]
            print(classification_report(all_true, all_preds, target_names=target_names))
            
            # Focus on problematic classes
            class4_true = np.array(all_true) == 4
            class4_pred = np.array(all_preds) == 4
            class9_true = np.array(all_true) == 9
            class9_pred = np.array(all_preds) == 9
            
            print(f"Class 4 (Gasriese) accuracy: {np.sum(class4_true & class4_pred) / (np.sum(class4_true) + 1e-10):.4f}")
            print(f"Class 9 (Hohestrahlung) accuracy: {np.sum(class9_true & class9_pred) / (np.sum(class9_true) + 1e-10):.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model for evaluation
    model.load_state_dict(best_model_state)

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best Acc: {best_val_acc:.4f}')
    plt.legend()
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.argmax(outputs, dim=1)
        pred_probs, _ = torch.max(probs, dim=1)
        all_preds = preds.cpu().numpy()
        pred_probs = pred_probs.cpu().numpy()

    final_acc = accuracy_score(y_val, all_preds)
    print("Final Validation Accuracy: {:.2f}%".format(final_acc * 100))

    # Generate detailed classification report
    target_names = [f"{i}: {class_names[i]}" for i in range(len(class_names))]
    print("\nClassification Report:")
    print(classification_report(y_val, all_preds, target_names=target_names))

    # Create and display confusion matrix
    conf_matrix = confusion_matrix(y_val, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Visualize embeddings with t-SNE
    with torch.no_grad():
        # Get embeddings from the penultimate layer
        feature_extractor = nn.Sequential(*list(model.model.children())[:-1])
        embeddings = feature_extractor(X_val_tensor).cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot with true labels
    plt.figure(figsize=(14, 12))
    # Create a custom colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    for i, color in enumerate(colors):
        mask = y_val == i
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[color], 
            label=class_names[i],
            alpha=0.7,
            s=50
        )
    plt.title("t-SNE of Validation Embeddings (True Labels)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('tsne_true_labels.png')
    plt.show()

    # Plot with predicted labels and confidence
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=all_preds, 
        cmap='viridis', 
        alpha=0.7,
        s=50*pred_probs  # Size points by confidence
    )
    plt.colorbar(scatter)
    plt.title("t-SNE of Validation Embeddings (Predicted Labels, size = confidence)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig('tsne_predicted_labels.png')
    plt.show()

    # Save the model, preprocessing objects, and feature engineering steps
    model_info = {
        'model_state_dict': model.state_dict(),
        'label_encoder': le,
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'output_dim': output_dim,
        'dropout_rate': dropout_rate,
        'feature_names': feature_names,
        'class_names': class_names
    }

    torch.save(model_info, "improved_cosmic_classifier_model.pth")
    print("Model and preprocessing objects saved.")
    
    return model, model_info, le

# Function to make predictions on new data
def predict_planet_class(model, new_data, model_info, scaler_path='planet_scaler.pkl'):
    """
    Predict planet class for new data.
    
    Args:
        model: PyTorch model
        new_data: DataFrame with the same columns as the training data (before preprocessing)
        model_info: Dictionary with model metadata
        scaler_path: Path to the saved scaler
    
    Returns:
        Tuple of (predicted_classes, class_names, confidences)
    """
    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    # Ensure we have the same raw features
    required_raw_features = ['Atmospheric_Density', 'Surface_Temperature', 'Gravity',
                          'Water_Content', 'Mineral_Abundance', 'Orbital_Period',
                          'Proximity_to_Star', 'Magnetic_Field_Strength', 'Radiation_Levels',
                          'Atmospheric_Composition_Index']
    
    # Check if we have all required features
    if not all(col in new_data.columns for col in required_raw_features):
        raise ValueError(f"Input data must have these columns: {required_raw_features}")
    
    # Create engineered features
    df = new_data.copy()
    
    # Radiation-related interactions
    df['radiation_temp_interaction'] = df['Radiation_Levels'] * df['Surface_Temperature']
    df['radiation_atmosphere_interaction'] = df['Radiation_Levels'] * df['Atmospheric_Composition_Index']
    df['radiation_magnetic_ratio'] = df['Radiation_Levels'] / (df['Magnetic_Field_Strength'] + 1e-5)

    # Gas giant related features
    df['density_gravity_ratio'] = df['Atmospheric_Density'] / (df['Gravity'] + 1e-5)
    df['orbital_proximity_ratio'] = df['Orbital_Period'] / (df['Proximity_to_Star'] + 1e-5)

    # Other potentially useful feature combinations
    df['water_temp_interaction'] = df['Water_Content'] * df['Surface_Temperature']
    df['mineral_gravity_product'] = df['Mineral_Abundance'] * df['Gravity']
    df['habitability_index'] = (df['Atmospheric_Composition_Index'] * 0.4 + 
                              (1 - abs(df['Surface_Temperature']/300)) * 0.3 +
                              (1 - df['Radiation_Levels']/10) * 0.3)

    # Additional engineering based on scientific intuition
    df['temperature_gravity_ratio'] = df['Surface_Temperature'] / (df['Gravity'] + 1e-5)
    df['water_atmosphere_product'] = df['Water_Content'] * df['Atmospheric_Composition_Index']
    df['mineral_proximity_ratio'] = df['Mineral_Abundance'] / (df['Proximity_to_Star'] + 1e-5)
    
    # Interaction terms for potential habitable worlds
    df['hab_potential'] = (df['Water_Content']/100) * df['Atmospheric_Composition_Index'] * (1-(abs(df['Surface_Temperature']-300)/300))
    
    # Desert planet indicators
    df['desert_indicator'] = (df['Surface_Temperature'] * (1 - df['Water_Content']/100))
    
    # Ice world indicators
    df['ice_indicator'] = ((1 - df['Surface_Temperature']/300) * df['Water_Content']/100)
    
    # Scale the data
    feature_names = model_info['feature_names']
    
    # Ensure all expected columns are present
    missing_cols = set(feature_names) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing engineered features: {missing_cols}")
    
    X = scaler.transform(df[feature_names].values)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.argmax(outputs, dim=1)
        confidences, _ = torch.max(probs, dim=1)
    
    predicted_classes = preds.cpu().numpy()
    class_names_list = [model_info['class_names'][cls] for cls in predicted_classes]
    confidences = confidences.cpu().numpy()
    
    return predicted_classes, class_names_list, confidences

if __name__ == "__main__":
    # Path to the preprocessed CSV file
    preprocessed_file = "cosmic-classifier-cogni25\preprocessed_planetary_data.csv"
    
    # Train the model
    model, model_info, label_encoder = train_model(preprocessed_file)
    
    print("\nExample usage of the prediction function:")
    print("To make predictions on new data:")
    print("predicted_classes, class_names, confidences = predict_planet_class(model, new_data, model_info)")