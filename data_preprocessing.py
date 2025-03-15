import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_and_save_planetary_data(input_filepath, output_filepath):
    """
    Preprocess planetary data and save to CSV
    
    Args:
        input_filepath: Path to the input CSV file
        output_filepath: Path to save the preprocessed data
    """
    print(f"Loading data from {input_filepath}...")
    # Load data
    data = pd.read_csv(input_filepath)
    print("Initial data shape:", data.shape)

    # Fix column names if needed
    expected_columns = ["Atmospheric_Density", "Surface_Temperature", "Gravity",
                        "Water_Content", "Mineral_Abundance", "Orbital_Period",
                        "Proximity_to_Star", "Magnetic_Field_Strength", "Radiation_Levels",
                        "Atmospheric_Composition_Index", "Planet_Class"]
                        
    if list(data.columns) != expected_columns:
        print("Fixing column names...")
        data.columns = expected_columns

    # Clean the data
    print("Removing missing values...")
    data = data.dropna()
    print("Data shape after dropping missing values:", data.shape)

    # Filter invalid labels
    def valid_label(x):
        if isinstance(x, (int, float)):
            return x >= 0
        return True

    data_clean = data[data["Planet_Class"].apply(valid_label)]
    print("Data shape after filtering invalid labels:", data_clean.shape)

    # Print class distribution
    class_counts = data_clean['Planet_Class'].value_counts().sort_index()
    print("Number of rows per class")
    for class_value, count in class_counts.items():
        print(f"Class {class_value}: {count}")

    # Extract features
    features = data_clean.drop("Planet_Class", axis=1)
    labels = data_clean["Planet_Class"]

    # Handle non-numeric columns if any
    non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print("Extracting numeric values from non-numeric columns:", non_numeric_cols.tolist())
        for col in non_numeric_cols:
            features[col] = features[col].str.extract('(\d+)').astype(float)

    # Feature engineering
    print("Adding engineered features...")
    
    # Radiation-related interactions
    features['radiation_temp_interaction'] = features['Radiation_Levels'] * features['Surface_Temperature']
    features['radiation_atmosphere_interaction'] = features['Radiation_Levels'] * features['Atmospheric_Composition_Index']
    features['radiation_magnetic_ratio'] = features['Radiation_Levels'] / (features['Magnetic_Field_Strength'] + 1e-5)

    # Gas giant related features
    features['density_gravity_ratio'] = features['Atmospheric_Density'] / (features['Gravity'] + 1e-5)
    features['orbital_proximity_ratio'] = features['Orbital_Period'] / (features['Proximity_to_Star'] + 1e-5)

    # Other potentially useful feature combinations
    features['water_temp_interaction'] = features['Water_Content'] * features['Surface_Temperature']
    features['mineral_gravity_product'] = features['Mineral_Abundance'] * features['Gravity']
    features['habitability_index'] = (features['Atmospheric_Composition_Index'] * 0.4 + 
                                    (1 - abs(features['Surface_Temperature'])/300) * 0.3 +  # Normalized temperature factor
                                    (1 - features['Radiation_Levels']/10) * 0.3)  # Normalized radiation factor

    # Additional engineering based on scientific intuition
    features['temperature_gravity_ratio'] = features['Surface_Temperature'] / (features['Gravity'] + 1e-5)
    features['water_atmosphere_product'] = features['Water_Content'] * features['Atmospheric_Composition_Index']
    features['mineral_proximity_ratio'] = features['Mineral_Abundance'] / (features['Proximity_to_Star'] + 1e-5)
    
    # Interaction terms for potential habitable worlds
    features['hab_potential'] = (features['Water_Content']/100) * features['Atmospheric_Composition_Index'] * (1-(abs(features['Surface_Temperature']-300)/300))
    
    # Desert planet indicators
    features['desert_indicator'] = (features['Surface_Temperature'] * (1 - features['Water_Content']/100))
    
    # Ice world indicators
    features['ice_indicator'] = ((1 - features['Surface_Temperature']/300) * features['Water_Content']/100)
    
    print("Feature set shape after engineering:", features.shape)

    # Scaling features
    print("Scaling features...")
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns
    )
    
    # Add labels back to create final dataframe
    preprocessed_data = features_scaled.copy()
    preprocessed_data['Planet_Class'] = labels.values
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, 'planet_scaler.pkl')
    print("Scaler saved to planet_scaler.pkl")
    
    # Save preprocessed data to CSV
    preprocessed_data.to_csv(output_filepath, index=False)
    print(f"Preprocessed data saved to {output_filepath}")
    
    return preprocessed_data

if __name__ == "__main__":
    # Change these paths to your actual file locations
    input_file = "cosmic-classifier-cogni25/data/cosmicclassifierTraining.csv"
    output_file = "preprocessed_planetary_data.csv"
    
    preprocessed_data = preprocess_and_save_planetary_data(input_file, output_file)
    print("Preprocessing complete!")