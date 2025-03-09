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

#######################
# Import Dependencies #
#######################

# Data science Librairies
import os
import numpy as np
import scipy.io
import pandas as pd

# Preprocess
from imblearn.over_sampling import SMOTE
import scipy.signal as sig

#### Deep learning utilities
os.environ["KERAS_BACKEND"] = "torch"# Set the Keras backend to PyTorch
import keras_core as keras #for Pytorch Backend
import torch #Torch function for GPU (A40)
from keras_core.utils import to_categorical

# Load Model
from model.CNN_BILSTM_keras import CNN_BISLTM


#Metrics
from sklearn.metrics import confusion_matrix
from utils.f1_score_function import f1_function

#############################################
# Train with Keras (Pytorch Backend) on GPU #
#############################################

print("Current Keras backend:", keras.backend.backend())

if torch.cuda.is_available() and torch.cuda.device_count() > 3:
    device = torch.device("cuda:3")  # Specify GPU 3
    print("Using GPU 3 for training")
else:
    device = torch.device("cpu")  # Use CPU if GPU 4 is not available
    print("GPU 3 is not available, using CPU")


#############
# Load data #
#############

# Train CSV
path = 'label/REFERENCE_train.csv'
file =  pd.read_csv(path, delimiter=',', header=None)
file.columns = ('filename','target')

mask = ~file["target"].str.contains('e') # Binary classification : 'O|~', E : correspond to nothing 

# Apply the mask to filter the data
filtered_filenames = file["filename"][mask]
filtered_targets = file["target"][mask]

# Test csv
path = 'label/REFERENCE_test.csv'
file =  pd.read_csv(path, delimiter=',', header=None)
file.columns = ('filename','target')

filtered_filenames_test = file["filename"][mask]
filtered_targets_test = file["target"][mask]
print(filtered_targets.shape,filtered_filenames.shape,filtered_filenames_test.shape,filtered_targets_test.shape)

########################
# Split data & Labeled #
########################

path_dataset_train = '../PhysionetAFDatabase/trainingSet'
path_dataset_val = '../PhysionetAFDatabase/validationSet'


#### Train Split & Label
X_train = filtered_filenames
X_test = filtered_filenames_test
y_train = filtered_targets

y_test= filtered_targets_test

train_frame ={'filename' : X_train, 'target': y_train}
pd_train = pd.DataFrame(train_frame) 
pd_train.groupby('target').describe()


val ={'filename' : X_test, 'target': y_test}
pd_val = pd.DataFrame(val) 
pd_val.groupby('target').describe()


a = np.array(X_train[:])
data_train = []
labels_train = []
data_RR = []
# Data : 11k to 2700 
A = []
for i in range(len(a)):
    AF = 0

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

    mat = scipy.io.loadmat(os.path.join(path_dataset_train,X_train.iloc[i]))
    data = np.array(mat['val'][:][:])
    data = data.flatten() 
    # Data clearn
    b,a = sig.butter(4,1,btype='highpass',fs = 300)
    data = sig.filtfilt(b, a,data)

    # Z-normalisation
    mean = np.mean(data)
    std = np.std(data)
    data  = (data-mean)/std
    
    
    shape = data.shape

    nb_slice_final = shape[0]
    nb_slice_begin = shape[0] - 9000
    
    part = np.arange(0, shape[0]+1, 3000)

    if shape[0] <= 3000:
        continue
    
    for j,sc_minus in enumerate(part):
        try:
            sc_plus = part[j+1]
        except:
            continue
        data_train.append(data[sc_minus:sc_plus])
        labels_train.append(AF)

train = np.asarray(data_train,'float32')
smote = SMOTE(sampling_strategy={3: 1500}, random_state=42)  # Specify desired samples for class 3
train, labels_train = smote.fit_resample(train, labels_train)
train = np.expand_dims(train, axis=-1)

class_counts = np.bincount(labels_train)

# Print the results
for class_label, count in enumerate(class_counts):
    print(f"Class {class_label}: {count} samples")
print(train.shape)



#### Test Split & Label
a = np.array(X_test[:])
data_test = []
labels_test = []
data_RR_test = []
# Data : 11k to 2700 
A = 0
N = 0
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
    
    mat = scipy.io.loadmat(os.path.join(path_dataset_val,X_test.iloc[i]))
    data = np.array(mat['val'][:][:])
    data = data.flatten()

    #data clearn
    b,a = sig.butter(4,1,btype='highpass',fs = 300)
    data = sig.filtfilt(b, a,data)

    # Z-normalisation
    mean = np.mean(data)
    std = np.std(data)
    data  = (data-mean)/std

    shape = data.shape
    nb_slice_final = shape[0]
    nb_slice_begin = shape[0] - 9000

    mean = np.mean(data)
    std = np.std(data)
    data  = (data-mean)/std
    
    if shape[0] >= 6000:
        labels_test.append(AF)
        data_test.append(data[3000:6000])
    elif shape[0] >= 3000:
        labels_test.append(AF)
        data_test.append(data[0:3000])
    else:
        continue
            

test = np.asarray(data_test,'float32')
test = np.expand_dims(data_test, axis=-1)


# One-hot Encoded
labels_train = to_categorical(labels_train, num_classes=4)
labels_test = to_categorical(labels_test, num_classes=4)

##############
# Load Model #
##############

model = CNN_BISLTM()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# List accumulation

result_loss = []
result_mae = []
matrix_result_test = []
f1_list = []

##################
# Prepare tensor #
##################

labels_test = np.argmax(labels_test, axis=1)

X_train,y_train = train,labels_train # Rename
X_train = torch.tensor(np.array(X_train), device=device, dtype=torch.float32)
y_train = torch.tensor(np.array(y_train), device=device, dtype=torch.float32)
test_tensor = torch.tensor(np.array(test), device=device, dtype=torch.float32)

# Train 
epoch = 90
model.fit(X_train.cpu().numpy(), y_train.cpu().numpy(),batch_size=2048,epochs=epoch)


# Predict
y_prediction = model.predict(test_tensor.cpu().numpy())
y_prediction = np.argmax(y_prediction, axis=1)
combined_confusion_matrix = confusion_matrix(labels_test, y_prediction)
f1_data = f1_function(combined_confusion_matrix)
matrix_result_test.append(combined_confusion_matrix)


############################
#  Save and Display Result #
############################

print(combined_confusion_matrix)
print(f1_data)

path = "../save_output"
os.makedirs(path, exist_ok=True)  # Creates the directory if it doesn't exist
file = '../save_output/matrix_result.npy'
np.save(file, matrix_result_test, allow_pickle=True)
file = '../save_output/f1_data_score.npy'
np.save(file, f1_data, allow_pickle=True)
