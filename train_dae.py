import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# ✅ Load Dataset
df = pd.read_csv("cosmic-classifier-cogni25\data\cosmicclassifierTraining.csv")  # Replace with actual dataset path

# ✅ Handle Missing Values (Fill or Drop)
df = df.fillna(df.mean(numeric_only=True))  # Fill NaN in numerical columns
df = df.dropna(subset=["Prediction"])  # Drop rows with missing target

# ✅ Identify Numerical & Categorical Columns
categorical_cols = [8, 9]  # Indices of categorical columns (Adjust if needed)
numerical_cols = list(set(df.columns) - set(categorical_cols) - {"Prediction"})  # Exclude target

# ✅ Preprocessing: Scale Numerical & Encode Categorical Data
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),  # Scale numerical columns
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)  # Encode categorical columns
])

# ✅ Extract Features & Target
X = df.drop(columns=["Prediction"])  # Features
y = df["Prediction"]  # Target

# ✅ Apply Preprocessing
X_processed = preprocessor.fit_transform(X)

# ✅ Check Shape after Preprocessing
print("Processed Feature Shape:", X_processed.shape)

# ✅ Introduce Noise (Denoising Autoencoder)
noise_factor = 0.2
X_noisy = X_processed + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_processed.shape)

# ✅ Define Autoencoder
input_dim = X_processed.shape[1]  # Number of features after encoding
encoding_dim = 32  # Size of compressed representation

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(encoding_dim, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)  # Output same shape as input

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# ✅ Train the Autoencoder
autoencoder.fit(X_noisy, X_processed, epochs=50, batch_size=32, shuffle=True, validation_split=0.1)

# ✅ Extract Features Using Encoder
encoder = Model(input_layer, encoded)
X_encoded = encoder.predict(X_processed)

# ✅ Check Shape of Encoded Features
print("Encoded Feature Shape:", X_encoded.shape)

# ✅ Save Encoded Features for Downstream Tasks
np.save("encoded_features.npy", X_encoded)
