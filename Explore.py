import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "missionstars_2024.11.18_15.55.38.csv"  # Update with your dataset path
data = pd.read_csv(file_path)

# Inspect the dataset
print("Dataset Columns:", data.columns)
print("Sample Data:\n", data.head())

# Define the target and features
target = "st_spttype"  # Spectral type
features = ["st_mass", "st_rad", "st_teff", "st_lbol", "st_vmag"]  # Adjust as needed

# Check if features and target exist in the dataset
missing_cols = [col for col in features + [target] if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# Handle missing values
data = data.dropna(subset=features + [target])  # Drop rows with missing target or features
print(f"Data shape after dropping missing values: {data.shape}")

# Encode target (Spectral Type) as numerical labels
data[target], class_mapping = pd.factorize(data[target])
print(f"Spectral Type Mapping: {dict(enumerate(class_mapping))}")

# Split the dataset
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_mapping, yticklabels=class_mapping)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance")
plt.show()