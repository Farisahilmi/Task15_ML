import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Perform data preprocessing steps:
    1. Drop unnecessary columns (student_id)
    2. Handle missing values
    3. Drop duplicates
    4. Encode categorical variables
    """
    # 1. Drop unnecessary columns
    if 'student_id' in df.columns:
        df = df.drop('student_id', axis=1)
        
    # 2. Handle missing values
    # For simplicity, we drop rows with missing values. 
    # (Alternatively, we can use imputation, but let's stick to dropping or simple fills)
    df = df.dropna()
    
    # 3. Drop duplicates
    df = df.drop_duplicates()
    
    # 4. Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # We will use LabelEncoder for all categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, label_encoders

def split_and_scale(df, target_col='burnout_level'):
    """Split into train and test sets, and scale features."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Train test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to dataframe for readability (optional, but good for keeping feature names)
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Test the preprocessing pipeline
    df = load_data('dataset/student_mental_health_burnout.csv')
    df_processed, encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_processed)
    print("Preprocessing completed successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
