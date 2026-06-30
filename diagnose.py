import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

# ======== 1. DATA OVERVIEW ========
df = pd.read_csv('dataset/student_mental_health_burnout.csv')
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# ======== 2. TARGET ANALYSIS ========
print("\n" + "=" * 60)
print("TARGET COLUMN: burnout_level")
print("=" * 60)
target = 'burnout_level'
print(f"Dtype: {df[target].dtype}")
print(f"Unique values: {df[target].unique()}")
print(f"\nValue counts:\n{df[target].value_counts()}")
print(f"\nValue counts (normalized):\n{df[target].value_counts(normalize=True).round(3)}")
n_classes = df[target].nunique()
print(f"\nNumber of classes: {n_classes}")
print(f"Baseline random accuracy (1/n_classes): {1/n_classes:.4f}")

# ======== 3. DATA QUALITY ========
print("\n" + "=" * 60)
print("DATA QUALITY")
print("=" * 60)
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# ======== 4. FEATURE ANALYSIS ========
print("\n" + "=" * 60)
print("FEATURE DETAILS")
print("=" * 60)
for col in df.columns:
    uv = df[col].unique()
    print(f"{col}: {df[col].nunique()} unique | dtype={df[col].dtype}")
    if df[col].dtype == 'object' and df[col].nunique() < 20:
        print(f"  -> Values: {sorted(list(uv))}")

# ======== 5. CORRELATION WITH TARGET ========
print("\n" + "=" * 60)
print("CORRELATION / RELATIONSHIP ANALYSIS")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder
df2 = df.copy()
for col in df2.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col].astype(str))

corr_with_target = df2.corr()[target].drop(target).sort_values(key=abs, ascending=False)
print("Correlation with burnout_level:")
print(corr_with_target)

# ======== 6. QUICK MODEL TEST ========
print("\n" + "=" * 60)
print("QUICK MODEL DIAGNOSTICS")
print("=" * 60)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

X = df2.drop(target, axis=1)
y = df2[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Raw RF without scaling
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Accuracy (no scaling): {acc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV scores: {cv_scores}")
print(f"CV mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\nTop 10 Feature Importances:\n{fi.head(10)}")

print("\nDONE")
