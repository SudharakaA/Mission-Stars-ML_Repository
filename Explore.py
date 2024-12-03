import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Load the dataset
file_path = "missionstars_2024.11.18_15.55.38.csv"  # Update with your dataset path
data = pd.read_csv(file_path)

# Display dataset summary
print("Dataset Columns:", data.columns)
print("Sample Data:\n", data.head())
print("Dataset Shape:", data.shape)

# Define target and features
target = "st_spttype"  # Target column (Spectral type)
features = [
    "st_mass", "st_rad", "st_teff", "st_lbol", "st_vmag",
    "st_dist", "st_logg", "st_age", "st_pmra", "st_pmdec"
]

# Check for missing features or target
missing_cols = [col for col in features + [target] if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}")

# Handle missing values
data = data.dropna(subset=features + [target])  # Drop rows with missing values
print(f"Data Shape After Dropping Missing Values: {data.shape}")

# Add Derived Features
data["mass_radius_ratio"] = data["st_mass"] / data["st_rad"]  # Derived feature
data["log_lbol"] = np.log1p(data["st_lbol"])  # Log transformation
features += ["mass_radius_ratio", "log_lbol"]

# Encode target (Spectral Type)
data[target], class_mapping = pd.factorize(data[target])
print(f"Spectral Type Mapping: {dict(enumerate(class_mapping))}")

# Split the dataset
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest Model ---
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# --- Neural Network Model ---
num_classes = len(class_mapping)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

nn_model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train, y_train_cat, epochs=20, batch_size=32, verbose=1, validation_split=0.2)
nn_y_pred = nn_model.predict(X_test).argmax(axis=1)

# --- Results and Visualizations ---
def save_and_display_results():
    # Classification Report
    rf_report = classification_report(y_test, rf_y_pred, output_dict=True)
    nn_report = classification_report(y_test, nn_y_pred, output_dict=True)

    # Save Reports
    pd.DataFrame(rf_report).transpose().to_csv("rf_classification_report.csv")
    pd.DataFrame(nn_report).transpose().to_csv("nn_classification_report.csv")

    # Confusion Matrices
    rf_cm = confusion_matrix(y_test, rf_y_pred)
    nn_cm = confusion_matrix(y_test, nn_y_pred)

    # Plot Confusion Matrices
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_mapping, yticklabels=class_mapping)
    plt.title("RF Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplot(1, 2, 2)
    sns.heatmap(nn_cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=class_mapping, yticklabels=class_mapping)
    plt.title("NN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrices.png")
    plt.show()

    # Feature Importance (RF only)
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv("feature_importance.csv")

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="coolwarm")
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    plt.show()

    # Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("feature_correlation_heatmap.png")
    plt.show()

    # Model Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("model_accuracy_plot.png")
    plt.show()

# --- User Interface ---
def on_predict():
    try:
        # Get user inputs
        input_features = {
            "st_mass": float(entry_mass.get()),
            "st_rad": float(entry_radius.get()),
            "st_teff": float(entry_temp.get()),
            "st_lbol": float(entry_lbol.get()),
            "st_vmag": float(entry_vmag.get()),
            "st_dist": float(entry_dist.get()),
            "st_logg": float(entry_logg.get()),
            "st_age": float(entry_age.get()),
            "st_pmra": float(entry_pmra.get()),
            "st_pmdec": float(entry_pmdec.get()),
            "mass_radius_ratio": float(entry_mass.get()) / float(entry_radius.get()),
            "log_lbol": np.log1p(float(entry_lbol.get()))
        }
        # Predict with Random Forest
        rf_pred = class_mapping[rf_model.predict(pd.DataFrame([input_features]))[0]]
        # Predict with Neural Network
        nn_pred = class_mapping[nn_model.predict(pd.DataFrame([input_features])).argmax()]
        # Display result
        messagebox.showinfo("Prediction Result",
                            f"Random Forest Prediction: {rf_pred}\nNeural Network Prediction: {nn_pred}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

def on_view_plots():
    save_and_display_results()
    messagebox.showinfo("Plots", "Graphs saved and displayed!")

# UI Creation
root = tk.Tk()
root.title("Star Classification Predictor")

fields = [
    ("Mass (st_mass)", "entry_mass"),
    ("Radius (st_rad)", "entry_radius"),
    ("Temperature (st_teff)", "entry_temp"),
    ("Luminosity (st_lbol)", "entry_lbol"),
    ("Apparent Magnitude (st_vmag)", "entry_vmag"),
    ("Distance (st_dist)", "entry_dist"),
    ("Log(g) (st_logg)", "entry_logg"),
    ("Age (st_age)", "entry_age"),
    ("Proper Motion RA (st_pmra)", "entry_pmra"),
    ("Proper Motion Dec (st_pmdec)", "entry_pmdec"),
]

entries = {}
for i, (label, var_name) in enumerate(fields):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[var_name] = entry

entry_mass = entries["entry_mass"]
entry_radius = entries["entry_radius"]
entry_temp = entries["entry_temp"]
entry_lbol = entries["entry_lbol"]
entry_vmag = entries["entry_vmag"]
entry_dist = entries["entry_dist"]
entry_logg = entries["entry_logg"]
entry_age = entries["entry_age"]
entry_pmra = entries["entry_pmra"]
entry_pmdec = entries["entry_pmdec"]

# Add Buttons
btn_predict = tk.Button(root, text="Predict", command=on_predict)
btn_predict.grid(row=len(fields), column=0, pady=10)

btn_view_plots = tk.Button(root, text="View Plots", command=on_view_plots)
btn_view_plots.grid(row=len(fields), column=1, pady=10)

root.mainloop()