import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
from tqdm import tqdm

# Boosting algorithms
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Load data
train_file = "cosmic-classifier-cogni25/data/cosmicclassifierTraining.csv"
data = pd.read_csv(train_file)
print("Initial data shape:", data.shape)

# Set correct column names if needed
expected_columns = ["Atmospheric_Density", "Surface_Temperature", "Gravity",
                   "Water_Content", "Mineral_Abundance", "Orbital_Period",
                   "Proximity_to_Star", "Magnetic_Field_Strength", "Radiation_Levels",
                   "Atmospheric_Composition_Index", "Planet_Class"]
if list(data.columns) != expected_columns:
    data.columns = expected_columns

# Preprocessing
data = data.dropna()
print("Data shape after dropping missing values:", data.shape)

# Filter invalid labels
def valid_label(x):
    if isinstance(x, (int, float)):
        return x >= 0
    return True
data_clean = data[data["Planet_Class"].apply(valid_label)]
print("Data shape after filtering invalid labels:", data_clean.shape)

# Display class distribution
class_counts = data_clean['Planet_Class'].value_counts()
print("Number of rows per class")
for class_value, count in class_counts.items():
    print(f"Class {class_value}: {count}")

# Create additional features through feature engineering
def add_engineered_features(df):
    """Add engineered features to potentially improve model performance"""
    df_new = df.copy()
    
    # Ensure numeric columns
    numeric_cols = ['Surface_Temperature', 'Gravity', 'Water_Content', 'Mineral_Abundance', 
                    'Radiation_Levels', 'Magnetic_Field_Strength', 'Proximity_to_Star', 'Orbital_Period']
    for col in numeric_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
    
    # Ratios and interactions
    df_new['Temp_Gravity_Ratio'] = df_new['Surface_Temperature'] / (df_new['Gravity'] + 1e-5)
    df_new['Water_Temp_Interaction'] = df_new['Water_Content'] * df_new['Surface_Temperature']
    df_new['Mineral_Water_Ratio'] = df_new['Mineral_Abundance'] / (df_new['Water_Content'] + 1e-5)
    df_new['Radiation_Magnetic_Ratio'] = df_new['Radiation_Levels'] / (df_new['Magnetic_Field_Strength'] + 1e-5)
    df_new['Proximity_Orbital_Ratio'] = df_new['Proximity_to_Star'] / (df_new['Orbital_Period'] + 1e-5)
    
    # Polynomial features for key attributes
    df_new['Gravity_Squared'] = df_new['Gravity'] ** 2
    df_new['Temp_Squared'] = df_new['Surface_Temperature'] ** 2
    
    # Log transformations for skewed features
    for col in ['Atmospheric_Density', 'Orbital_Period', 'Magnetic_Field_Strength']:
        df_new[f'Log_{col}'] = np.log1p(np.abs(df_new[col]))
    
    # Combined indicators
    df_new['Habitability_Index'] = (
        df_new['Atmospheric_Composition_Index'] * 0.4 +
        (1 - df_new['Radiation_Levels']) * 0.3 +
        df_new['Water_Content'] * 0.3
    )
    
    # Binned features
    df_new['Temp_Bins'] = pd.qcut(df_new['Surface_Temperature'], 5, labels=False, duplicates='drop')
    df_new['Gravity_Bins'] = pd.qcut(df_new['Gravity'], 5, labels=False, duplicates='drop')
    
    return df_new

# Apply feature engineering
data_engineered = add_engineered_features(data_clean)
print(f"Original features: {data_clean.shape[1]-1}, With engineered features: {data_engineered.shape[1]-1}")

# Prepare features
features = data_engineered.drop("Planet_Class", axis=1)

# Handle non-numeric columns if any
non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print("Extracting numeric values from non-numeric columns:", non_numeric_cols.tolist())
    for col in non_numeric_cols:
        features[col] = features[col].str.extract('(\d+)').astype(float)
print("Transformed feature set first 5 columns:", features.iloc[:, :5].head())

# Prepare data for modeling
X = features.values
y = data_engineered["Planet_Class"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoded classes:", le.classes_)

# Try different scalers to find the optimal one
def evaluate_scalers():
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'PowerTransformer': PowerTransformer()
    }
    
    results = {}
    X_train_base, X_test_base, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    for name, scaler in scalers.items():
        X_train = scaler.fit_transform(X_train_base)
        X_test = scaler.transform(X_test_base)
        
        # Quick evaluation with a baseline model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        results[name] = accuracy
        print(f"Scaler: {name}, Accuracy: {accuracy:.4f}")
    
    best_scaler_name = max(results, key=results.get)
    print(f"Best scaler: {best_scaler_name} with accuracy {results[best_scaler_name]:.4f}")
    return scalers[best_scaler_name]

# Find the best scaler
print("\nEvaluating different scalers...")
best_scaler = evaluate_scalers()

# Scale features with the best scaler
X_scaled = best_scaler.fit_transform(X)

# Feature selection with RFECV
def select_features(X_train, y_train, X_val):
    print("\nPerforming feature selection with RFECV...")
    selector = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy',
        min_features_to_select=5,
        n_jobs=-1
    )
    
    selector.fit(X_train, y_train)
    print(f"Optimal number of features: {selector.n_features_}")
    
    # Get selected feature indices
    feature_indices = np.where(selector.support_)[0]
    
    # Transform data
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    
    return X_train_selected, X_val_selected, feature_indices

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print("Training samples:", X_train.shape[0], "Validation samples:", X_val.shape[0])

# Apply feature selection
X_train_selected, X_val_selected, selected_indices = select_features(X_train, y_train, X_val)
print(f"Selected {len(selected_indices)} features out of {X_train.shape[1]}")

# Get names of selected features
selected_features = features.columns[selected_indices].tolist()
print("Selected features:", selected_features)

# Function to evaluate model
def evaluate_model(model, model_name, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n{model_name} Results:")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Classification report
    target_names = [str(x) for x in le.classes_]
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    return model, accuracy

# Hyperparameter optimization with Optuna for XGBoost
def optimize_xgboost(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_float('alpha', 0, 5),
        'lambda': trial.suggest_float('lambda', 0, 5),
        'random_state': 42
    }
    
    model = XGBClassifier(**param)
    
    # Use cross-validation to evaluate
    cv_scores = cross_val_score(
        model, X_train_selected, y_train, 
        cv=StratifiedKFold(n_splits=5), 
        scoring='accuracy'
    )
    
    return np.mean(cv_scores)

# Hyperparameter optimization for LightGBM
def optimize_lightgbm(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }
    
    model = LGBMClassifier(**param)
    
    # Use cross-validation to evaluate
    cv_scores = cross_val_score(
        model, X_train_selected, y_train, 
        cv=StratifiedKFold(n_splits=5), 
        scoring='accuracy'
    )
    
    return np.mean(cv_scores)

# Hyperparameter optimization for CatBoost
def optimize_catboost(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        'random_seed': 42,
        'verbose': False
    }
    
    model = CatBoostClassifier(**param)
    
    # Use cross-validation to evaluate
    cv_scores = cross_val_score(
        model, X_train_selected, y_train, 
        cv=StratifiedKFold(n_splits=5), 
        scoring='accuracy'
    )
    
    return np.mean(cv_scores)

# Run optimization for XGBoost
print("\nOptimizing XGBoost hyperparameters with Optuna...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(optimize_xgboost, n_trials=20)
print("Best XGBoost trial:")
print(f"  Value: {study_xgb.best_value:.4f}")
print(f"  Params: {study_xgb.best_params}")

# Run optimization for LightGBM
print("\nOptimizing LightGBM hyperparameters with Optuna...")
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(optimize_lightgbm, n_trials=20)
print("Best LightGBM trial:")
print(f"  Value: {study_lgbm.best_value:.4f}")
print(f"  Params: {study_lgbm.best_params}")

# Run optimization for CatBoost
print("\nOptimizing CatBoost hyperparameters with Optuna...")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(optimize_catboost, n_trials=20)
print("Best CatBoost trial:")
print(f"  Value: {study_cat.best_value:.4f}")
print(f"  Params: {study_cat.best_params}")

# Create optimized models
xgb_model = XGBClassifier(**study_xgb.best_params)
lgbm_model = LGBMClassifier(**study_lgbm.best_params)
cat_model = CatBoostClassifier(**study_cat.best_params, verbose=False)

# Define other base models
rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
hgb_model = HistGradientBoostingClassifier(max_iter=200, max_depth=8, random_state=42)

# Train and evaluate individual models
print("\n=== Training and evaluating individual models ===")
models = {
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "CatBoost": cat_model,
    "RandomForest": rf_model,
    "HistGradientBoosting": hgb_model
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    trained_model, accuracy = evaluate_model(model, name, X_train_selected, X_val_selected, y_train, y_val)
    results[name] = accuracy
    trained_models[name] = trained_model

# Create voting ensemble
print("\n=== Creating Voting Ensemble ===")
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', trained_models["XGBoost"]),
        ('lgbm', trained_models["LightGBM"]),
        ('cat', trained_models["CatBoost"]),
        ('rf', trained_models["RandomForest"]),
        ('hgb', trained_models["HistGradientBoosting"])
    ],
    voting='soft'
)

voting_model, voting_acc = evaluate_model(
    voting_clf, "Voting Ensemble", 
    X_train_selected, X_val_selected, y_train, y_val
)
results["Voting Ensemble"] = voting_acc

# Create stacking ensemble
print("\n=== Creating Stacking Ensemble ===")
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(**study_xgb.best_params)),
        ('lgbm', LGBMClassifier(**study_lgbm.best_params)),
        ('cat', CatBoostClassifier(**study_cat.best_params, verbose=False)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking_model, stacking_acc = evaluate_model(
    stacking_clf, "Stacking Ensemble", 
    X_train_selected, X_val_selected, y_train, y_val
)
results["Stacking Ensemble"] = stacking_acc

# Compare model performances
print("\n=== Model Performance Comparison ===")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

# Identify best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Plot model comparisons
plt.figure(figsize=(12, 6))
models_names = list(results.keys())
accuracies = [results[name] for name in models_names]

# Sort by accuracy
sorted_indices = np.argsort(accuracies)
sorted_names = [models_names[i] for i in sorted_indices]
sorted_accs = [accuracies[i] for i in sorted_indices]

bars = plt.barh(sorted_names, sorted_accs, color='skyblue')
plt.xlabel('Validation Accuracy')
plt.title('Model Performance Comparison')
plt.xlim(min(accuracies) - 0.05, 1.0)

# Add accuracy values at the end of each bar
for i, v in enumerate(sorted_accs):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# Final evaluation with best model
best_model = None
if best_model_name == "Voting Ensemble":
    best_model = voting_model
elif best_model_name == "Stacking Ensemble":
    best_model = stacking_model
else:
    best_model = trained_models[best_model_name]

print("\n=== Final Evaluation of Best Model ===")
y_pred = best_model.predict(X_val_selected)
target_names = [str(x) for x in le.classes_]

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=target_names))

# Plot confusion matrix
plt.figure(figsize=(12, 10))
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig("best_model_confusion_matrix.png")
plt.close()

# Save pipeline components
print("\nSaving final model and components...")
joblib.dump(best_model, "cosmic_best_model.pkl")
joblib.dump(best_scaler, "cosmic_scaler.pkl")
joblib.dump(selected_indices, "cosmic_selected_features.pkl")
joblib.dump(le, "cosmic_label_encoder.pkl")

print("\nCreating inference pipeline and sample prediction code...")
# Create a sample prediction function
def predict_planet_class(data_sample, model, scaler, selected_indices, label_encoder, feature_names):
    """Make predictions on new data samples"""
    # Ensure data is in the right format (DataFrame with proper feature names)
    if isinstance(data_sample, pd.DataFrame):
        # Apply feature engineering
        sample_engineered = add_engineered_features(data_sample)
        # Extract features in the correct order
        sample_features = sample_engineered[feature_names].values
    else:
        # Assuming data_sample is a numpy array or similar
        sample_features = data_sample
    
    # Scale the features
    sample_scaled = scaler.transform(sample_features)
    
    # Select only the relevant features
    sample_selected = sample_scaled[:, selected_indices]
    
    # Make prediction
    prediction_encoded = model.predict(sample_selected)
    
    # Convert encoded prediction back to original class names
    prediction = label_encoder.inverse_transform(prediction_encoded)
    
    return prediction

# Save sample code for inference
sample_code = """
import joblib
import pandas as pd
import numpy as np

# Load the saved components
model = joblib.load("cosmic_best_model.pkl")
scaler = joblib.load("cosmic_scaler.pkl")
selected_indices = joblib.load("cosmic_selected_features.pkl")
label_encoder = joblib.load("cosmic_label_encoder.pkl")

def add_engineered_features(df):
    \"\"\"Add engineered features to potentially improve model performance\"\"\"
    df_new = df.copy()
    
    # Ratios and interactions
    df_new['Temp_Gravity_Ratio'] = df_new['Surface_Temperature'] / (df_new['Gravity'] + 1e-5)
    df_new['Water_Temp_Interaction'] = df_new['Water_Content'] * df_new['Surface_Temperature']
    df_new['Mineral_Water_Ratio'] = df_new['Mineral_Abundance'] / (df_new['Water_Content'] + 1e-5)
    df_new['Radiation_Magnetic_Ratio'] = df_new['Radiation_Levels'] / (df_new['Magnetic_Field_Strength'] + 1e-5)
    df_new['Proximity_Orbital_Ratio'] = df_new['Proximity_to_Star'] / (df_new['Orbital_Period'] + 1e-5)
    
    # Polynomial features for key attributes
    df_new['Gravity_Squared'] = df_new['Gravity'] ** 2
    df_new['Temp_Squared'] = df_new['Surface_Temperature'] ** 2
    
    # Log transformations for skewed features
    for col in ['Atmospheric_Density', 'Orbital_Period', 'Magnetic_Field_Strength']:
        df_new[f'Log_{col}'] = np.log1p(np.abs(df_new[col]))
    
    # Combined indicators
    df_new['Habitability_Index'] = (
        df_new['Atmospheric_Composition_Index'] * 0.4 +
        (1 - df_new['Radiation_Levels']) * 0.3 +
        df_new['Water_Content'] * 0.3
    )
    
    # Binned features
    df_new['Temp_Bins'] = pd.qcut(df_new['Surface_Temperature'], 5, labels=False, duplicates='drop')
    df_new['Gravity_Bins'] = pd.qcut(df_new['Gravity'], 5, labels=False, duplicates='drop')
    
    return df_new

def predict_planet_class(data_sample):
    \"\"\"Make predictions on new planetary data\"\"\"
    # Apply feature engineering
    sample_engineered = add_engineered_features(data_sample)
    
    # Extract all features (including engineered ones)
    sample_features = sample_engineered.values
    
    # Scale the features
    sample_scaled = scaler.transform(sample_features)
    
    # Select only the relevant features
    sample_selected = sample_scaled[:, selected_indices]
    
    # Make prediction
    prediction_encoded = model.predict(sample_selected)
    
    # Convert encoded prediction back to original class names
    prediction = label_encoder.inverse_transform(prediction_encoded)
    
    return prediction

# Example usage
# test_data = pd.read_csv("cosmic-classifier-cogni25/data/test_planets.csv")
# predictions = predict_planet_class(test_data)
# print(predictions)
"""

with open("cosmic_inference_code.py", "w") as f:
    f.write(sample_code)

print("Complete! The optimized model and inference code have been saved.")