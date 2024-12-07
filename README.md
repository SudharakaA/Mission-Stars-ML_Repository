# Star Classification System

## Project Overview
This project is a **Star Classification System** that uses machine learning models to classify stars based on their physical and astronomical properties. The system leverages a dataset containing features such as mass, radius, temperature, luminosity, and more to predict the spectral type of a star using Random Forest and Neural Network models. It also provides a user-friendly interface for prediction and visualization.

---

## Features
- **Machine Learning Models**:
  - Random Forest Classifier: Offers interpretable feature importance.
  - Neural Network Classifier: Provides higher flexibility and accuracy for complex relationships.
- **Visualization**:
  - Confusion matrices for model evaluation.
  - Feature importance for Random Forest.
  - Correlation heatmap for input features.
  - Training accuracy vs. validation accuracy plot for Neural Network.
- **User Interface**:
  - A Tkinter-based GUI allows users to input star attributes and get predictions.
- **Derived Features**:
  - `mass_radius_ratio`: A ratio of mass to radius.
  - `log_lbol`: Log transformation of luminosity.

---

## Dataset
- **File**: `missionstars_2024.11.18_15.55.38.csv`
- **Target Variable**: `st_spttype` (Spectral Type)
- **Input Features**:
  - `st_mass`: Mass of the star.
  - `st_rad`: Radius of the star.
  - `st_teff`: Effective temperature.
  - `st_lbol`: Luminosity.
  - `st_vmag`: Apparent magnitude.
  - `st_dist`: Distance from Earth.
  - `st_logg`: Logarithm of surface gravity.
  - `st_age`: Age of the star.
  - `st_pmra`: Proper motion in RA.
  - `st_pmdec`: Proper motion in Dec.

---

## Prerequisites

### Python Libraries
Install the following libraries before running the code:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `seaborn`
- `matplotlib`
- `tkinter` (built-in in most Python installations)

To install missing libraries, run:
```bash
pip install pandas numpy scikit-learn tensorflow seaborn matplotlib
```

### Dataset
Ensure the dataset file (`missionstars_2024.11.18_15.55.38.csv`) is present in the same directory as the script.

---

## How to Run
1. **Load the Dataset**:
   - Place your dataset file in the script’s directory.
   - Ensure the dataset includes the required columns listed in the "Dataset" section.

2. **Execute the Script**:
   ```bash
   python star_classification.py
   ```

3. **User Interface**:
   - Launches a GUI where users can input star properties and get predictions from both Random Forest and Neural Network models.

4. **View Results**:
   - The script saves classification reports, confusion matrices, feature importance plots, and accuracy plots in the same directory.

---

## Key Functions

### Data Preprocessing
- **Missing Values**: Rows with missing values are dropped.
- **Feature Engineering**:
  - `mass_radius_ratio`: Derived as `st_mass / st_rad`.
  - `log_lbol`: Logarithm of `st_lbol`.

### Model Training
- **Random Forest**:
  - A 100-tree Random Forest is trained on 80% of the dataset.
- **Neural Network**:
  - A 3-layer model with ReLU activations and dropout regularization.
  - Trained using categorical cross-entropy loss and Adam optimizer.

### Predictions
- The GUI accepts user inputs and uses both models to predict the spectral type of a star.

### Visualizations
- **Confusion Matrices**: Shows the performance of both models.
- **Feature Importance**: Highlights influential features for the Random Forest model.
- **Correlation Heatmap**: Displays relationships between input features.
- **Accuracy Plot**: Visualizes training and validation accuracy over epochs for the Neural Network.

---

## Output Files
- **Classification Reports**:
  - `rf_classification_report.csv`
  - `nn_classification_report.csv`
- **Plots**:
  - `confusion_matrices.png`
  - `feature_importance.png`
  - `feature_correlation_heatmap.png`
  - `model_accuracy_plot.png`

---

## User Interface
- **Prediction**:
  - Enter star properties into the input fields.
  - Click the "Predict" button to view predictions from both models.
- **Visualization**:
  - Click the "View Plots" button to save and display plots of results.

---

## Limitations
- **Dataset Dependence**: The accuracy depends on the quality and quantity of the dataset.
- **Model Selection**: Random Forest is interpretable but may underperform compared to Neural Networks in certain cases.
- **User Input**: GUI accepts numeric inputs only; invalid entries may cause errors.

---

## Future Improvements
- Enhance the Neural Network architecture for better accuracy.
- Incorporate cross-validation for more robust model evaluation.
- Automate data cleaning and preprocessing steps.
- Add error handling in the GUI to improve user experience.

---

## License
This project is open-source and available under the MIT License.

---

## Authors
Developed by a collaborative team passionate about astrophysics and machine learning.





🧠 Technologies Used 
        •       Programming Language: Python
	•	Libraries: OpenCV, TensorFlow/Keras, NumPy, Matplotlib
	•	Model Architectures: Feedforward Neural Networks (FNNs), UNet

 📊 How It Works
        1.	Data Preprocessing: Clean and normalize starfield images, enhancing visibility of faint stars.
	2.	Model Training: Train a FNN on labeled star patterns to learn spatial relationships and features.
	3.	Pattern Matching: Compare detected patterns to predefined constellations for classification.


 🌌 Applications
        •	Astronomy: Automatic identification of celestial patterns in large datasets.
	•	Space Navigation: Use as a star sensor for spacecraft orientation.
	•	Education: A tool for amateur astronomers to explore and learn about the night sky.

 🤝 Contributors
        •	Sudharaka Ashen
	•	Deshan Lanka
	•	Ravindu Yasas

 🛰️ Future Enhancements
        •	Expand the dataset to include more constellations and celestial objects.
	•	Incorporate real-time detection using video feeds.
	•	Improve detection accuracy for noisy or low-resolution images.
![Screenshot 2024-12-03 at 20 18 15](https://github.com/user-attachments/assets/96819c79-bc41-47b2-9fa4-e391c2760580)


![shooting-star-the-smurfs](https://github.com/user-attachments/assets/42d3992a-4478-4838-a196-8974236a073a)

