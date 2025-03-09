# Physionet Challenge 2017 : Atrial Fibrilation Classification on single lead ECG

### Author : **Sane Viola**
### Summary :
- Download Dataset 
- Train Deep Learning model on Keras with Pytorch Backend
- Train Deep Learning model on Keras with Tensorflow Backend
- Train a Random Forest model 

### Short explaination :
##### Pathophysiology :
Atrial fibrillation (AF) is the most common arrhythmia, with a global prevalence of 1%. ECG findings are crucial for diagnosing AF. Over the past decade, the advent of smartwatches has introduced a new approach to healthcare: personalized medicine, enabling individuals to monitor their health. ECG measurements on smartwatches are comparable to lead I of the traditional 12-lead ECG.

ECG findings :
- Irregularly irregular rythm
- Absence of P-waves
- QRS complex < 120 ms
- Variable Ventricular rates 

### Assumption :
**Test Dataset Assumption** : Due to the fact that Test dataset is not leak, Validation correspond to the test dataset
**Project Assumption** : _If a cardiologist can diagnose atrial fibrillation, deep learning can replicate this by leveraging relevant patterns_
**Training Assumption** : Three Python files are available: one containing a machine learning model, and the other two implementing deep learning models. Both deep learning models use Keras, but with different backends.



### Download Dataset :
The training set contains 8,528 single lead ECG recordings, between 9s to 60s of temporal signal. 
4 classes are available : 
- **Normal Class**: Represents a normal heart rhythm without arrhythmia, observed on a single lead ECG.
- **Atrial Fibrillation Class**: Represents arrhythmia corresponding to the ECG findings described above, specifically atrial fibrillation.
- **Other Arrhythmia Class**: Includes various arrhythmias that cannot be specifically classified (e.g., Atrial Flutter, BAV, Cardiac Stroke, etc.). While the exact type is difficult to determine, it is crucial to classify these events to achieve good sensitivity in detection.
- **Noise Class**: Refers to different types of interference, such as movement noise, high heart rate, or other artifacts that can affect the ECG signal.


[**Dataset Challenge Here**](https://physionet.org/content/challenge-2017/1.0.0/)

<img width="800" alt="image" src="https://github.com/user-attachments/assets/7b3ae52f-0de3-4361-98ce-474010b25e56" />


### Deep Learning :
Train on GPU
Train on CPU
Train with Keras and tensorflow Backend
Train with Keras and pytorch Backend

##### Architecture : 
<img width="1400" alt="image" src="https://github.com/user-attachments/assets/dccbb574-8d94-4ece-bb5f-7ae08caad2f7" />


### Result : 

**F1-Overall** : **0.946** 

**10-Fold Cross validation F1-Overall**: 0.896 +/- 0.026 [0,852 ; 0,925] 

Confusion Matrix : 

<img width="373" alt="image" src="https://github.com/user-attachments/assets/8bac8cc2-672f-405a-aec3-39be9fce1969" />


