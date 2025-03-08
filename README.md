<img width="336" alt="image" src="https://github.com/user-attachments/assets/ee2058a4-51d7-482b-8663-99f01c753ccc" /># Physionet Challenge 2017 : Atrial Fibrilation Classification on single lead ECG

### Summary :
- Download Dataset
- Train Deep Learning model --> Slow 
- Train Random Forest model --> Rapid Test 
- Test model

### Short explaination :
###### Pathophysiology :
Atrial fibrilation(AF) is the most common arrhythmia with 1% of prevalences in Worldwide. 
The ECG findings is the most important diagnostic to observe a AF. 
In the last decade, a new approach of medicine permit by the arrive of smartwaches : Personalized medecine, permit to people to survey their heath.
ECG on smartwatches are comparable to the lead I in traditional twelve leads ECG. 

ECG findings :
- Irregularly irregular rythm
- Absence of P-waves
- QRS complex < 120 ms
- Variable Ventricular rates 

### Assumption :
**Test Dataset Assumption** : Due to the fact that Test dataset is not leak, Validation correspond to the test dataset
**Project Assumption** : _If a cardiologist can diagnose atrial fibrillation, deep learning can replicate this by leveraging relevant patterns_



### Download Dataset :
The training set contains 8,528 single lead ECG recordings, between 9s to 60s of temporal signal. 
4 classes are available : 
- Normal class : non arrythmia on a signle lead 
[**Dataset Challenge Here**](https://physionet.org/content/challenge-2017/1.0.0/)
<img width="414" alt="image" src="https://github.com/user-attachments/assets/7b3ae52f-0de3-4361-98ce-474010b25e56" />


### Deep Learning :
Train on GPU
Train on CPU
Train with Keras and tensorflow Backend
Train with Keras and pytorch Backend

##### Architecture : 
<img width="1638" alt="image" src="https://github.com/user-attachments/assets/dccbb574-8d94-4ece-bb5f-7ae08caad2f7" />


### Result : 

**F1-Overall** : **0.946** 
**10-Fold Cross validation F1-Overall**: 0.896 +/- 0.026 [0,852 ; 0,925] 

Confusion Matrix : 
<img width="373" alt="image" src="https://github.com/user-attachments/assets/8bac8cc2-672f-405a-aec3-39be9fce1969" />


