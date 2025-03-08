
import numpy as np 
from sklearn.metrics import confusion_matrix

def f1_function(combined_confusion_matrix):


    sum_N_predicted = np.sum(combined_confusion_matrix[:,0])
    sum_A_predicted = np.sum(combined_confusion_matrix[:,1])
    sum_other_predicted = np.sum(combined_confusion_matrix[:,2])
    sum_noise_predicted =np.sum(combined_confusion_matrix[:,3])

    sum_N_true = np.sum(combined_confusion_matrix[0,:])
    sum_A_true = np.sum(combined_confusion_matrix[1,:])
    sum_other_true= np.sum(combined_confusion_matrix[2,:])
    sum_noise_true =np.sum(combined_confusion_matrix[3,:])
    
    f1_N = 2*combined_confusion_matrix[0,0]/(sum_N_predicted+sum_N_true)
    f1_A = 2*combined_confusion_matrix[1,1]/(sum_A_predicted+sum_A_true)

    f1_other = 2*combined_confusion_matrix[2,2]/(sum_other_predicted+sum_other_true)
    f1_noise = 2*combined_confusion_matrix[3,3]/(sum_noise_predicted+sum_noise_true)

    f1 = (f1_noise+f1_A+f1_N+f1_other)/4

    f1_data = [f1_N,f1_A,f1_other,f1_noise,f1]

    return f1_data