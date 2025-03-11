import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 🔹 Step 1: Load Dataset
df = pd.read_csv("cosmic-classifier-cogni25\data\cosmicclassifierTraining.csv")  # Replace with actual filename

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=["number"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns  # Object means categorical

# Handle missing values:
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))  # Median for numbers
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))  # Mode for categories
# 🔹 Step 2: Handle Missing Values
df.fillna(df.median(), inplace=True)  # Fill missing values with median

# 🔹 Step 3: Handle Outliers (Using IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Clip outliers to acceptable range
df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)

# 🔹 Step 4: Scale Features (Standardization)
scaler = StandardScaler()
X = df.drop(columns=["planet_type"])  # Drop target column
X_scaled = scaler.fit_transform(X)

# 🔹 Step 5: Encode Target Labels (German Planet Types)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["planet_type"])

# 🔹 Step 6: Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 Step 7: Build MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(128, 96, 64, 32), activation='relu',
                     solver='adam', max_iter=500, random_state=42)

# 🔹 Step 8: Train Model
mlp.fit(X_train, y_train)

# 🔹 Step 9: Predict on Test Data
y_pred = mlp.predict(X_test)

# 🔹 Step 10: Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print("🔹 MLP Accuracy:", accuracy)
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
