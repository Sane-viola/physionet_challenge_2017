
#!/usr/bin/env python
# coding: utf-8

"""
Author: Sane Viola 
Github : https://github.com/Sane-viola
Linkedin : www.linkedin.com/in/sane-viola

Date: 2025-03-09
Description: Classification on single lead ECG, database from Physionet challenge 2017 : https://physionet.org/content/challenge-2017/1.0.0/


Summary:
1. Import Dependencies – Load required libraries for data processing, deep learning, and evaluation.
2. Check GPU Availability – Configure the script to run on GPU if available; otherwise, use CPU.
3. Load Data – Read training and testing datasets from CSV files.
4. Preprocess Data – Filter, normalize, and segment ECG signals.
5. Apply Data Augmentation – Use SMOTE to balance the dataset.
6. Prepare Data for Training – Convert data into tensors and apply one-hot encoding.
7. Load Model – Initialize and compile the CNN-BiLSTM model.
8. Train Model – Train the model using the prepared dataset.
9. Evaluate Model – Predict test labels and compute performance metrics.
10. Save and Display Results – Store the confusion matrix and F1 score.

"""

# Import necessary libraries
#### Import
import numpy as np  
import os  
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

# Metric 
from sklearn.metrics import confusion_matrix 
# Model 
from sklearn.ensemble import RandomForestClassifier # Random Forest model
# preprocessing
from imblearn.over_sampling import SMOTE  # Handling class imbalance
import pywt  # wavelet transformations
import scipy.io  # loading MATLAB files
import neurokit2 as nk  #  ECG signal processing


# Load training data from CSV file
path = '/Users/sane/Library/Mobile Documents/com~apple~CloudDocs/Master 2 IHU LIRYC/APP project/APP 1 - Atrial Fibrillation/Data/PhysionetAFDatabase/TrainingSet/REFERENCE.csv'
file = pd.read_csv(path, delimiter=',', header=None)
file.columns = ('filename', 'target')

# Filter out unwanted classes for classification
mask = ~file["target"].str.contains('e')  # Exclude records with class 'e', # Classification : 'A|N|O|~'
filtered_filenames = file["filename"][mask]
filtered_targets = file["target"][mask]

# Load test data
path = '/Users/sane/Library/Mobile Documents/com~apple~CloudDocs/Master 2 IHU LIRYC/APP project/APP 1 - Atrial Fibrillation/Data/PhysionetAFDatabase/validationSet/REFERENCE.csv'
file = pd.read_csv(path, delimiter=',', header=None)
file.columns = ('filename', 'target')

# Apply the same filtering to test data
filtered_filenames_test = file["filename"][mask]
filtered_targets_test = file["target"][mask]
print(filtered_targets.shape, filtered_filenames.shape, filtered_filenames_test.shape, filtered_targets_test.shape)

# Split data into training and testing sets
X_train = filtered_filenames
X_test = filtered_filenames_test
y_train = filtered_targets
y_test = filtered_targets_test

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Summarize the training data
train_frame = {'filename': X_train, 'target': y_train}
pd_train = pd.DataFrame(train_frame)
pd_train.groupby('target').describe()

# Summarize the test data
val = {'filename': X_test, 'target': y_test}
pd_val = pd.DataFrame(val)
pd_val.groupby('target').describe()

# Preprocess training data
a = np.array(X_train[:])
data_train = []  # To store preprocessed training signals
labels_train = []  # To store corresponding labels
data_RR = []  # Placeholder for additional feature
for i in range(len(a)):
    AF = 0  # Default class

    #  target labels to numerical values
    if y_train.iloc[i] == 'N':
        AF = 0
    elif y_train.iloc[i] == 'A':
        AF = 1
    elif y_train.iloc[i] == 'O':
        AF = 2
    elif y_train.iloc[i] == '~':
        AF = 3
    else:
        print('5 classes error')
        break

    # Load and preprocess ECG data
    mat = scipy.io.loadmat(os.path.join('Data/PhysionetAFDatabase/TrainingSet', X_train.iloc[i]))
    data = np.array(mat['val'][:][:]).flatten()
    data = nk.ecg_clean(data, sampling_rate=300, method='pantompkins1985')  # Clean ECG data

    # Standardize the data
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std

    # Slice the data into overlapping windows
    shape = data.shape
    part = np.arange(0, shape[0] + 1, 3000)

    # Skip signals with less than 3000 samples
    if shape[0] <= 3000:
        continue

    for j in range(len(part) - 1):
        sc_minus = part[j]
        sc_plus = part[j + 1]
        data_train.append(data[sc_minus:sc_plus])
        labels_train.append(AF)

# Convert training data to NumPy arrays and handle class imbalance with SMOTE
train = np.asarray(data_train, 'float32')
smote = SMOTE(sampling_strategy={3: 1500}, random_state=42, k_neighbors=10)  # Oversample class 3
train, labels_train = smote.fit_resample(train, labels_train)
print(train.shape)
print(len(labels_train))

# Preprocess test data similarly
a = np.array(X_test[:])
data_test = []  # To store preprocessed test signals
labels_test = []  # To store corresponding test labels
for i in range(len(a)):
    if y_test.iloc[i] == 'N':
        AF = 0
    elif y_test.iloc[i] == 'A':
        AF = 1
    elif y_test.iloc[i] == 'O':
        AF = 2
    elif y_test.iloc[i] == '~':
        AF = 3
    else:
        print('5 classes error')
        break

    # Load and preprocess ECG data
    mat = scipy.io.loadmat(os.path.join('Data/PhysionetAFDatabase/validationSet', X_test.iloc[i]))
    data = np.array(mat['val'][:][:]).flatten()

    # Clean the ECG data
    data = nk.ecg_clean(data, sampling_rate=300, method='pantompkins1985')

    # Standardize the data
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std

    # Slice test data into windows of size 3000
    if shape[0] >= 6000:
        labels_test.append(AF)
        data_test.append(data[3000:6000])
    elif shape[0] >= 3000:
        labels_test.append(AF)
        data_test.append(data[0:3000])
    else:
        continue

print(f'Shape of data test is: {np.array(data_test).shape}')
test = np.asarray(data_test, 'float32')

#  function for feature extraction using wavelet decomposition
def extract_features(data):
    features = []
    for sig in data:
        coeffs = pywt.wavedec(sig, 'db4', level=4)  # Wavelet decomposition
        mean_features = [np.expand_dims(np.mean(c), axis=0) for c in coeffs]  # Compute mean of coefficients
        features.append(np.concatenate(mean_features))
    return np.array(features)

features = extract_features(train)  # Extract features for training data

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=32, verbose=2, n_jobs=-1)
model.fit(features, labels_train)

# Predict on the test set
y_prediction = model.predict(extract_features(test))

# Compute the confusion matrix
combined_confusion_matrix = confusion_matrix(labels_test, y_prediction)

# Calculate F1 scores for each class
sum_N_predicted = np.sum(combined_confusion_matrix[:, 0])
sum_A_predicted = np.sum(combined_confusion_matrix[:, 1])
sum_other_predicted = np.sum(combined_confusion_matrix[:, 2])
sum_noise_predicted = np.sum(combined_confusion_matrix[:, 3])

sum_N_true = np.sum(combined_confusion_matrix[0, :])
sum_A_true = np.sum(combined_confusion_matrix[1, :])
sum_other_true = np.sum(combined_confusion_matrix[2, :])
sum_noise_true = np.sum(combined_confusion_matrix[3, :])

f1_N = 2 * combined_confusion_matrix[0, 0] / (sum_N_predicted + sum_N_true)
f1_A = 2 * combined_confusion_matrix[1, 1] / (sum_A_predicted + sum_A_true)
f1_other = 2 * combined_confusion_matrix[2, 2] / (sum_other_predicted + sum_other_true)
f1_noise = 2 * combined_confusion_matrix[3, 3] / (sum_noise_predicted + sum_noise_true)

f1 = (f1_noise + f1_A + f1_N + f1_other) / 4

# Print F1 scores and plot confusion matrix
print(f1_N, f1_N, f1_other, f1_noise, f1)
plt.figure(figsize=(8, 6))
sns.heatmap(combined_confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'AF', 'Other', 'Noisy'], 
            yticklabels=['Normal', 'AF', 'Other', 'Noisy'])
plt.title("Confusion Matrix for All Classes")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
