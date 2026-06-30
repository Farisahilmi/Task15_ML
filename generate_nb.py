import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

markdown_1 = """
# Exploratory Data Analysis (EDA) - Student Burnout Prediction
Notebook ini berisi proses eksplorasi data, preprocessing, dan pelatihan model Random Forest untuk dataset `student_burnout.csv`.
"""

code_1 = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
"""

markdown_2 = """
## 1. Data Overview
Memuat dataset dan melihat informasi awalnya.
"""

code_2 = """\
df = pd.read_csv('dataset/student_burnout.csv')
print(f"Dataset Shape: {df.shape}")
df.head()
"""

code_3 = """\
df.info()
"""

markdown_3 = """
## 2. Preprocessing Data & Feature Engineering
Membuat label target `burnout_level` (Low/Medium/High) dari `burnout_score` (1-5).
"""

code_4 = """\
def map_score(score):
    if score <= 2:
        return 'Low'
    elif score == 3:
        return 'Medium'
    else:
        return 'High'

df['burnout_level'] = df['burnout_score'].apply(map_score)

# Melihat distribusi target baru
print(df['burnout_level'].value_counts())
sns.countplot(data=df, x='burnout_level', order=['Low', 'Medium', 'High'], palette='viridis')
plt.title('Distribusi Level Burnout')
plt.show()
"""

markdown_4 = """
### Menangani Missing Values dan Kolom Tidak Penting
"""

code_5 = """\
# Drop kolom yang tidak digunakan sebagai fitur
cols_to_drop = ['student_id', 'burnout_score', 'high_burnout']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Drop missing values
df = df.dropna().drop_duplicates()
print(f"Shape setelah cleaning: {df.shape}")
"""

markdown_5 = """
## 3. Exploratory Data Analysis (EDA)
Melihat hubungan antara beberapa fitur kunci dengan tingkat burnout.
"""

code_6 = """\
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='burnout_level', y='sleep_hours', order=['Low', 'Medium', 'High'], palette='Set2')
plt.title('Hubungan Jam Tidur dengan Level Burnout')
plt.show()
"""

code_7 = """\
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='burnout_level', y='homework_hours', order=['Low', 'Medium', 'High'], palette='Set3')
plt.title('Hubungan Jam PR/Belajar dengan Level Burnout')
plt.show()
"""

code_8 = """\
# Encoding categorical data for correlation heatmap
from sklearn.preprocessing import LabelEncoder
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

plt.figure(figsize=(12, 10))
sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Korelasi Antar Fitur')
plt.show()
"""

markdown_6 = """
## 4. Modeling - Random Forest Classifier
Melatih model Random Forest dan mengevaluasi akurasinya.
"""

code_9 = """\
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df_encoded.drop('burnout_level', axis=1)
y = df_encoded['burnout_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Prediksi
y_pred = rf.predict(X_test_scaled)

# Hitung Akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"\\n✅ Random Forest Accuracy: {accuracy * 100:.2f}%\\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))
"""

code_10 = """\
# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features[indices], palette='viridis')
plt.title('Feature Importances - Random Forest')
plt.xlabel('Relative Importance')
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(markdown_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(markdown_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(markdown_5),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_code_cell(code_8),
    nbf.v4.new_markdown_cell(markdown_6),
    nbf.v4.new_code_cell(code_9),
    nbf.v4.new_code_cell(code_10),
]

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
