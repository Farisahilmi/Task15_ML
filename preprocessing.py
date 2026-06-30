import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATASET_PATH = 'dataset/student_burnout.csv'

def load_data(file_path=DATASET_PATH):
    """Load dataset dari file CSV."""
    df = pd.read_csv(file_path)
    return df

def create_burnout_label(df):
    """
    Buat kolom 'burnout_level' (Low/Medium/High) dari 'burnout_score' (1–5):
      - Low    : score 1–2
      - Medium : score 3
      - High   : score 4–5
    """
    def map_score(score):
        if score <= 2:
            return 'Low'
        elif score == 3:
            return 'Medium'
        else:
            return 'High'
    df = df.copy()
    df['burnout_level'] = df['burnout_score'].apply(map_score)
    return df

def preprocess_data(df):
    """
    Lakukan tahapan preprocessing:
    1. Drop kolom yang tidak diperlukan (student_id, burnout_score, high_burnout)
    2. Buat label burnout_level dari burnout_score
    3. Handle missing values
    4. Drop duplikat
    5. Encode kolom kategorik (gender)
    """
    # 1. Buat label target sebelum drop kolom
    df = create_burnout_label(df)

    # 2. Drop kolom tidak diperlukan
    cols_to_drop = ['student_id', 'burnout_score', 'high_burnout']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # 3. Handle missing values
    df = df.dropna()

    # 4. Drop duplikat
    df = df.drop_duplicates()

    # 5. Encode kolom kategorik
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def split_and_scale(df, target_col='burnout_level'):
    """Split data ke train/test set dan lakukan scaling fitur."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Train-test split 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisasi fitur numerik
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_df, X_test_df, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_data()
    df_processed, encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_processed)
    print("Preprocessing selesai.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")
