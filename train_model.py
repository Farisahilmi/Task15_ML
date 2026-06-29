import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from preprocessing import load_data, preprocess_data, split_and_scale

def train_and_evaluate():
    print("Loading data...")
    df = load_data('dataset/student_mental_health_burnout.csv')
    
    print("Preprocessing data...")
    df_processed, encoders = preprocess_data(df)
    
    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_processed)
    
    # We use RandomForestClassifier because:
    # 1. It handles non-linear relationships well.
    # 2. It is robust to outliers compared to other models like Logistic Regression.
    # 3. It provides feature importance natively, which helps in analyzing factors affecting burnout.
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    feature_importances = rf_model.feature_importances_
    features = X_train.columns
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(importance_df.head())
    
    # Save the model, scaler, and encoders
    print("\nSaving model and preprocessing objects...")
    model_data = {
        'model': rf_model,
        'scaler': scaler,
        'encoders': encoders,
        'features': features
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Saved as 'model.pkl'. Training completed.")

if __name__ == "__main__":
    train_and_evaluate()
