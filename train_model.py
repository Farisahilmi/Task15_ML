import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score

from preprocessing import load_data, preprocess_data, split_and_scale, FEATURE_COLUMNS


def train_and_evaluate():
    print("=" * 60)
    print("STUDENT BURNOUT PREDICTION — RANDOM FOREST TRAINING")
    print("=" * 60)

    print("\n[1/5] Memuat data...")
    df = load_data()
    print(f"  Dataset shape: {df.shape}")

    print("\n[2/5] Preprocessing data...")
    df_processed, encoders = preprocess_data(df)
    print(f"  Setelah preprocessing: {df_processed.shape}")
    print(f"  Distribusi kelas target:\n{df_processed['burnout_level'].value_counts()}")

    print("\n[3/5] Split dan scaling data...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_processed)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # RandomForestClassifier dipilih karena:
    # 1. Robust terhadap outlier dan non-linearitas
    # 2. Tidak memerlukan asumsi distribusi data
    # 3. Memberikan feature importance secara native
    # 4. Ensemble dari banyak decision tree → prediksi lebih stabil
    print("\n[4/5] Melatih Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("  Training selesai.")

    print("\n[5/5] Evaluasi model...")
    y_pred = rf_model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n--- Metrik Evaluasi ---")
    print(f"  Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1-Score : {f1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("--- Confusion Matrix ---")
    le_target = encoders.get('burnout_level')
    if le_target is not None:
        int_labels = list(range(len(le_target.classes_)))
        str_labels = list(le_target.classes_)   # e.g. ['High', 'Low', 'Medium']
        cm = confusion_matrix(y_test, y_pred, labels=int_labels)
        print(pd.DataFrame(cm, index=str_labels, columns=[f'Pred {l}' for l in str_labels]))
    else:
        int_labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=int_labels)
        str_labels = [str(l) for l in int_labels]
        print(pd.DataFrame(cm, index=str_labels, columns=[f'Pred {l}' for l in str_labels]))

    # 5-Fold Cross Validation
    print("\n--- 5-Fold Cross Validation ---")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV Mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature Importance
    features = X_train.columns
    fi = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Top 10 Feature Importances ---")
    for feat, imp in fi.head(10).items():
        bar = "#" * int(imp * 100)
        print(f"  {feat:<28} {imp:.4f} {bar}")

    # Simpan model
    print("\nMenyimpan model ke 'model.pkl'...")
    model_data = {
        'model'               : rf_model,
        'scaler'              : scaler,
        'encoders'            : encoders,
        'features'            : features,
        'metrics'             : {
            'accuracy' : accuracy,
            'precision': precision,
            'recall'   : recall,
            'f1'       : f1,
            'cv_mean'  : cv_scores.mean(),
            'cv_std'   : cv_scores.std(),
        },
        'confusion_matrix'    : cm,
        'confusion_labels'    : str_labels,
        'feature_importances' : fi.to_dict(),
        'class_labels'        : str_labels,
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("=" * 60)
    print("  Model berhasil disimpan sebagai 'model.pkl'")
    print("  Training selesai!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
