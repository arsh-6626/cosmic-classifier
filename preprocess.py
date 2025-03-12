import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Function to load and explore the dataset
def load_and_explore_data(filepath):
    """
    Load the dataset and perform initial exploration
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Display basic information
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Data types and basic statistics
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

# Function to drop rows with NaN values
def drop_nan_rows(df, target_column='PlanetType'):
    """
    Drop rows that contain NaN values in either the target column or any feature
    """
    # Count rows before dropping
    rows_before = len(df)
    
    # Drop rows with NaN in the target column
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])
        rows_after_target = len(df)
        print(f"Dropped {rows_before - rows_after_target} rows with NaN in target column '{target_column}'")
    
    # # Drop rows with NaN in any feature
    # rows_after_target = len(df)
    # df = df.dropna()
    # rows_after = len(df)
    # print(f"Dropped {rows_after_target - rows_after} rows with NaN in feature columns")
    
    # Total rows dropped
    total_dropped = rows_before 
    print(f"Total rows dropped: {total_dropped} ({total_dropped/rows_before*100:.2f}% of data)")
    
    return df
# Function to preprocess categorical variables
def encode_categorical_features(df, target_column=None):
    """
    One-hot encode categorical variables in the dataset
    """
    # Create a copy to avoid modifying the original dataframe
    df_encoded = df.copy()
    
    # Identify categorical columns (excluding the target if it's categorical)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    print(f"Categorical columns to encode: {categorical_columns}")
    
    # If there are categorical columns to encode
    if categorical_columns:
        # Use pandas get_dummies for one-hot encoding
        df_encoded = pd.get_dummies(
            df_encoded, 
            columns=categorical_columns,
            drop_first=False,  # Keep all dummy columns
            prefix_sep='_',
            dummy_na=False     # Don't create dummies for NaN values
        )
        print(f"After encoding, dataframe has {df_encoded.shape[1]} columns")
    else:
        print("No categorical columns to encode")
    
    return df_encoded
def convert_to_numeric(df):
    """
    Convert string columns that should be numeric to float
    """
    numeric_columns = df.columns.drop([col for col in df.columns if df[col].dtype == 'object'])
    
    for col in numeric_columns:
        try:
            # Try to convert to numeric, with errors='coerce' to handle non-convertible values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Converted {col} to numeric")
        except Exception as e:
            print(f"Error converting {col}: {e}")
    
    return df

# Function for data visualization
def visualize_data(df, target_column='PlanetType'):
    """
    Create visualizations for better understanding of the data
    """
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    
    # Histogram of all numerical features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    # Check if there are any numeric features to plot
    if numeric_features:
        for i, feature in enumerate(numeric_features):
            plt.subplot(3, 4, i+1)
            plt.hist(df[feature], bins=20)
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()
        
        plt.savefig('feature_distributions.png')
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation = df[numeric_features].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        # Box plots for detecting outliers
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(numeric_features):
            plt.subplot(3, 4, i+1)
            sns.boxplot(y=df[feature])
            plt.title(f'Boxplot of {feature}')
            plt.tight_layout()
        
        plt.savefig('boxplots.png')
        plt.close()
    else:
        print("No numeric features found for visualization")
    
    # If target is categorical, visualize class distribution
    if target_column in df.columns and df[target_column].dtype == 'object':
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[target_column])
        plt.title('Class Distribution')
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()
        
        # Feature distributions by class (only if there are numeric features)
        if numeric_features:
            for feature in numeric_features[:3]:  # Just first 3 features to save space
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[target_column], y=df[feature])
                plt.title(f'{feature} by Planet Type')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(f'{feature}_by_class.png')
                plt.close()

# Function to clean the data
def clean_data(df, target_column='PlanetType'):
    """
    Handle missing values, outliers, and inconsistencies
    """
    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Step 1: Handle missing values
    numeric_features = cleaned_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    # Impute missing values in numeric columns with median (more robust than mean)
    for col in numeric_features:
        if cleaned_df[col].isnull().sum() > 0:
            median_val = cleaned_df[col].median()
            cleaned_df[col].fillna(median_val, inplace=True)
    
    # For categorical columns, fill with mode
    categorical_features = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_features:
        if cleaned_df[col].isnull().sum() > 0:
            mode_val = cleaned_df[col].mode()[0]
            cleaned_df[col].fillna(mode_val, inplace=True)
    
    # Step 2: Handle outliers using z-score method
    # Define a threshold for z-score (commonly 3)
    z_threshold = 3
    
    # Create a dictionary to store outlier information
    outlier_info = {}
    
    for col in numeric_features:
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(cleaned_df[col]))
        
        # Identify outliers
        outliers = (z_scores > z_threshold)
        outlier_count = np.sum(outliers)
        
        if outlier_count > 0:
            outlier_info[col] = outlier_count
            
            # Cap outliers at z_threshold * std from the mean (winsorization)
            mean_val = cleaned_df[col].mean()
            std_val = cleaned_df[col].std()
            
            # Cap upper outliers
            upper_limit = mean_val + z_threshold * std_val
            cleaned_df.loc[cleaned_df[col] > upper_limit, col] = upper_limit
            
            # Cap lower outliers
            lower_limit = mean_val - z_threshold * std_val
            cleaned_df.loc[cleaned_df[col] < lower_limit, col] = lower_limit
    
    print("Outliers capped for these features:", outlier_info)
    
    return cleaned_df

# Function for feature engineering
def engineer_features(df):
    """
    Create new features and transform existing ones
    """
    # Create a copy to avoid modifying the original dataframe
    engineered_df = df.copy()
    
    # Get list of available columns to avoid errors
    available_cols = engineered_df.columns.tolist()
    
    # Step 1: Create interaction features based on scientific intuition
    new_features_added = []
    
    if all(col in available_cols for col in ['Surface Temperature', 'Proximity to Star']):
        try:
            engineered_df['Temp_Distance_Ratio'] = engineered_df['Surface Temperature'] / (engineered_df['Proximity to Star'] + 0.001)
            new_features_added.append('Temp_Distance_Ratio')
        except Exception as e:
            print(f"Error creating Temp_Distance_Ratio: {e}")
    
    if all(col in available_cols for col in ['Gravity', 'Atmospheric Density']):
        try:
            engineered_df['Gravity_Density_Ratio'] = engineered_df['Gravity'] / (engineered_df['Atmospheric Density'] + 0.001)
            new_features_added.append('Gravity_Density_Ratio')
        except Exception as e:
            print(f"Error creating Gravity_Density_Ratio: {e}")
    
    # Step 2: Create a habitability index
    if all(col in available_cols for col in ['Water Content', 'Surface Temperature', 'Atmospheric Composition Index']):
        try:
            # Normalize Temperature for habitability (assuming 300K is optimal, and using a Gaussian-like function)
            temp_factor = np.exp(-((engineered_df['Surface Temperature'] - 300) / 100) ** 2)
            
            # Combine water, temperature factor, and atmospheric composition
            engineered_df['Habitability_Index'] = (
                0.4 * (engineered_df['Water Content'] / 100) + 
                0.3 * temp_factor + 
                0.3 * engineered_df['Atmospheric Composition Index']
            )
            new_features_added.append('Habitability_Index')
        except Exception as e:
            print(f"Error creating Habitability_Index: {e}")
    
    # Step 3: Create a resource value index
    if all(col in available_cols for col in ['Mineral Abundance', 'Gravity']):
        try:
            # Higher mineral abundance and lower gravity make mining easier
            engineered_df['Resource_Value'] = engineered_df['Mineral Abundance'] / np.sqrt(engineered_df['Gravity'] + 0.001)
            new_features_added.append('Resource_Value')
        except Exception as e:
            print(f"Error creating Resource_Value: {e}")
    
    # Step 4: Create a radiation safety index
    if all(col in available_cols for col in ['Radiation Levels', 'Magnetic Field Strength']):
        try:
            # Higher magnetic field protects from radiation
            engineered_df['Radiation_Safety'] = 1 / (1 + engineered_df['Radiation Levels'] / (engineered_df['Magnetic Field Strength'] + 0.001))
            new_features_added.append('Radiation_Safety')
        except Exception as e:
            print(f"Error creating Radiation_Safety: {e}")
    
    # Step 5: Apply log transformation to features with likely skewed distributions
    skewed_features = ['Atmospheric Density', 'Orbital Period', 'Proximity to Star']
    for feature in skewed_features:
        if feature in available_cols:
            try:
                # Add a small constant to avoid log(0)
                engineered_df[f'Log_{feature}'] = np.log1p(engineered_df[feature])
                new_features_added.append(f'Log_{feature}')
            except Exception as e:
                print(f"Error creating Log_{feature}: {e}")
    
    print("Engineered features added:", new_features_added)
    
    return engineered_df

# Function for feature scaling
def scale_features(df, target_column='PlanetType', method='standard'):
    """
    Scale features using various methods
    method: 'standard', 'minmax', or 'robust'
    """
    # Create a copy to avoid modifying the original dataframe
    scaled_df = df.copy()
    
    # Identify numerical columns (exclude the target)
    numeric_features = scaled_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    # Check if there are numeric features to scale
    if not numeric_features:
        print("Warning: No numeric features found for scaling")
        return scaled_df, None
    
    # Choose the appropriate scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    # Apply scaling to numeric features
    scaled_df[numeric_features] = scaler.fit_transform(scaled_df[numeric_features])
    
    return scaled_df, scaler
def handle_nan_values(X):
    """
    Handle NaN values in the feature matrix
    """
    # Check for NaN values
    if X.isna().any().any():
        print(f"Warning: Found {X.isna().sum().sum()} NaN values in the data")
        
        # Fill NaN values with column means
        X = X.fillna(X.mean())
        
        # If any columns are all NaN, fill with zeros
        X = X.fillna(0)
        
    return X
# Function for feature selection
def select_features(X, y, k=8):
    """
    Select top k features using ANOVA F-value with NaN handling
    """
    # Check if k is greater than the number of features
    if k > X.shape[1]:
        k = X.shape[1]
        print(f"Warning: k was reduced to {k} (the number of available features)")
    
    # Check for NaN values
    nan_count = np.isnan(X).sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in the data")
        
        # Fill NaN values with column means
        if isinstance(X, pd.DataFrame):
            X_filled = X.fillna(X.mean())
        else:
            # If X is numpy array
            X_filled = np.nan_to_num(X, nan=np.nanmean(X))
            
        print("NaN values have been filled with column means")
    else:
        X_filled = X
    
    # Create a selector
    selector = SelectKBest(f_classif, k=k)
    
    # Apply feature selection
    X_selected = selector.fit_transform(X_filled, y)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    
    return X_selected, selected_indices, selector

# Function to run PCA for dimensionality reduction
def apply_pca(X, n_components=5):
    """
    Apply PCA to reduce dimensionality
    """
    # Check if n_components is greater than the number of features
    if n_components > X.shape[1]:
        n_components = X.shape[1]
        print(f"Warning: n_components was reduced to {n_components} (the number of features)")
    
    # Create a PCA object
    pca = PCA(n_components=n_components)
    
    # Apply PCA
    X_pca = pca.fit_transform(X)
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("Explained variance by component:", explained_variance)
    print("Cumulative explained variance:", cumulative_variance)
    
    return X_pca, pca

# Function to prepare the final dataset for modeling
def prepare_for_modeling(df, target_column='PlanetType', test_size=0.2, random_state=42):
    """
    Split the data and prepare it for modeling
    """
    # Check if the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in the dataframe")
    
    # Check for NaN values and fill them
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in the dataframe before splitting")
        # Fill NaN values with column means
        df = df.fillna(df.mean())
        print("NaN values have been filled with column means")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

# Complete preprocessing pipeline
def preprocess_planetary_data(filepath, target_column='PlanetType', scaling_method='robust', 
                             apply_feature_selection=True, k_best_features=8,
                             apply_dimensionality_reduction=False, n_components=5):
    """
    Complete preprocessing pipeline for planetary classification
    """
    try:
        # Step 1: Load and explore data
        print("Step 1: Loading and exploring data...")
        df = load_and_explore_data(filepath)
        
        # Step 1.5: Convert string columns to numeric
        print("\nStep 1.5: Converting string columns to numeric...")
        df = convert_to_numeric(df)
        
         # Step 1.6: Drop rows with NaN values
        print("\nStep 1.6: Dropping rows with NaN values...")
        df = drop_nan_rows(df, target_column)
        
        # Step 2: Visualize data
        print("\nStep 2: Visualizing data...")
        visualize_data(df, target_column)
        
        # Step 3: Clean data
        print("\nStep 3: Cleaning data...")
        cleaned_df = clean_data(df, target_column)
         # Step 3.5: One-hot encode categorical features
        print("\nStep 3.5: One-hot encoding categorical features...")
        encoded_df = encode_categorical_features(cleaned_df, target_column)
        
        # Step 4: Engineer features
        print("\nStep 4: Engineering features...")
        engineered_df = engineer_features(encoded_df)
        
        # Step 5: Scale features
        print("\nStep 5: Scaling features...")
        scaled_df, scaler = scale_features(engineered_df, target_column, method=scaling_method)
        
        # Step 6: Prepare for modeling
        print("\nStep 6: Preparing for modeling...")
        X_train, X_test, y_train, y_test = prepare_for_modeling(scaled_df, target_column)
        
        # Store all feature names
        feature_names = X_train.columns.tolist()
        
        # Optional: Feature selection
        if apply_feature_selection:
            print("\nApplying feature selection...")
            X_train_selected, selected_indices, selector = select_features(X_train, y_train, k=k_best_features)
            X_test_selected = selector.transform(X_test)
            
            # Get names of selected features
            selected_features = [feature_names[i] for i in selected_indices]
            print(f"Selected features: {selected_features}")
            
            X_train = X_train_selected
            X_test = X_test_selected
        
        # Optional: Apply PCA
        if apply_dimensionality_reduction:
            print("\nApplying PCA...")
            X_train_pca, pca = apply_pca(X_train, n_components=n_components)
            X_test_pca = pca.transform(X_test)
            
            X_train = X_train_pca
            X_test = X_test_pca
        
        # Return the preprocessed data
        preprocessing_results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': feature_names,
            'original_df': df,
            'cleaned_df': cleaned_df,
            'engineered_df': engineered_df,
            'scaled_df': scaled_df
        }
        
        if apply_feature_selection:
            preprocessing_results['selector'] = selector
            preprocessing_results['selected_features'] = selected_features
        
        if apply_dimensionality_reduction:
            preprocessing_results['pca'] = pca
        
        print("\nPreprocessing complete!")
        return preprocessing_results
    
    except Exception as e:
        print(f"Error in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None
# Add this function to your existing code
def save_preprocessed_data(results, output_dir="."):
    """
    Save the preprocessed training and test data to CSV files
    
    Parameters:
    results (dict): The dictionary returned by preprocess_planetary_data
    output_dir (str): Directory where to save the CSV files
    """
    import os
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert selected features data to DataFrames
    if isinstance(results['X_train'], np.ndarray):
        # If X_train is a numpy array (after feature selection or PCA)
        if 'selected_features' in results:
            # Use selected feature names if available
            feature_names = results['selected_features']
        else:
            # Otherwise, create generic feature names
            feature_names = [f'feature_{i}' for i in range(results['X_train'].shape[1])]
        
        X_train_df = pd.DataFrame(results['X_train'], columns=feature_names)
        X_test_df = pd.DataFrame(results['X_test'], columns=feature_names)
    else:
        # If X_train is already a DataFrame
        X_train_df = results['X_train']
        X_test_df = results['X_test']
    
    # Add target variable to the DataFrames
    train_df = X_train_df.copy()
    test_df = X_test_df.copy()
    
    # Get target column name
    if isinstance(results['y_train'], pd.Series):
        target_name = results['y_train'].name
    else:
        target_name = 'PlanetType'
    
    train_df[target_name] = results['y_train']
    test_df[target_name] = results['y_test']
    
    # Save to CSV
    train_path = os.path.join(output_dir, 'preprocessed_train.csv')
    test_path = os.path.join(output_dir, 'preprocessed_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    
    # Optionally, save a complete version of the preprocessed data
    if 'scaled_df' in results:
        full_path = os.path.join(output_dir, 'full_preprocessed_data.csv')
        results['scaled_df'].to_csv(full_path, index=False)
        print(f"Full preprocessed data saved to: {full_path}")

        
# Example usage
if __name__ == "__main__":
    # Replace 'planetary_data.csv' with your actual file path
    results = preprocess_planetary_data(
        'cosmic-classifier-cogni25\data\cosmicclassifierTraining.csv',
        target_column='Prediction',
        scaling_method='robust',
        apply_feature_selection=True,
        k_best_features=8,
        apply_dimensionality_reduction=False
    )
    
    if results:
        # Access the preprocessed data
        X_train = results['X_train']
        X_test = results['X_test']
        y_train = results['y_train']
        y_test = results['y_test']
        
        print(f"\nFinal training set shape: {X_train.shape}")
        print(f"Final test set shape: {X_test.shape}")

        save_preprocessed_data(results, output_dir="./preprocessed_data")