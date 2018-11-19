##################################### Importing the libraries #########################################
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from imblearn.datasets import make_imbalance
########################################### Dataset ##################################################
#Generating the datasets (Blob)
X_blob, y_blob = make_blobs(n_samples=20000,n_features=2, centers=2)
y_blob=y_blob.reshape(20000,1)
dataset_blob= np.concatenate((X_blob,y_blob),axis=1)
df_blob=pd.DataFrame(dataset_blob,columns=('feature_1','feature_2','labels'))

#PLotting the dataset
sn.FacetGrid(data=df_blob,hue='labels',size=3).map(plt.scatter,'feature_1','feature_2')
plt.legend()
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()

#Generating imbalance data from the given dataset
X_res, y_res = make_imbalance(X_blob, y_blob,sampling_strategy={0:5, 1:10},ratio={1:30},random_state=42)
y_res=y_res.reshape(10030,1)
dataset_imb_blob= np.concatenate((X_res,y_res),axis=1)
df_imb_blob=pd.DataFrame(dataset_imb_blob,columns=('feature_1','feature_2','labels'))

#PLotting the imbalanced dataset
sn.FacetGrid(data=df_imb_blob,hue='labels',size=3).map(plt.scatter,'feature_1','feature_2')
plt.legend()
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()

################################## Techniques to handle imbalanced dataset ###########################
'''
 Technique 1:Undersampling
 The technique is implemented when there is millions of data and constrained resources such as
 limited memory , computational power
'''
from imblearn.under_sampling import RandomUnderSampler 
rus = RandomUnderSampler(random_state=42)
X1_res, y1_res = rus.fit_sample(X_res, y_res)
y1_res=y1_res.reshape(60,1)
dataset_blob1= np.concatenate((X1_res,y1_res),axis=1)
df_blob1=pd.DataFrame(dataset_blob1,columns=('feature_1','feature_2','labels'))

#PLotting the imbalanced dataset
sn.FacetGrid(data=df_blob1,hue='labels',size=3).map(plt.scatter,'feature_1','feature_2')
plt.legend()
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()

'''
Technique 2:Oversmapling
The technique is implemented when we have few sample of one class and want to create more samples
so we can balance the data
'''
from imblearn.over_sampling import RandomOverSampler
rus1 = RandomOverSampler(ratio='minority',random_state=42)
X2_res, y2_res = rus1.fit_sample(X_res, y_res)
y2_res=y2_res.reshape(20000,1)
dataset_blob2= np.concatenate((X2_res,y2_res),axis=1)
df_blob2=pd.DataFrame(dataset_blob2,columns=('feature_1','feature_2','labels'))

print('The number of samples of class 0 and 1:',pd.value_counts(df_blob2['labels'].values, sort=False))

#PLotting the imbalanced dataset
sn.FacetGrid(data=df_blob2,hue='labels',size=3).map(plt.scatter,'feature_1','feature_2')
plt.legend()
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()

'''
Technique 3:Creating synthetic features (SMOTE)
Creates new synthetic features rather than just repeating the features as compared to 
traditional oversampling 
'''
from imblearn.over_sampling import SMOTE
rus3 = SMOTE(ratio='minority',k_neighbors=3,random_state=42)
X3_res, y3_res = rus3.fit_sample(X_res, y_res)
y3_res=y3_res.reshape(20000,1)
dataset_blob3= np.concatenate((X3_res,y3_res),axis=1)
df_blob3=pd.DataFrame(dataset_blob3,columns=('feature_1','feature_2','labels'))

print('The number of samples of class 0 and 1:',pd.value_counts(df_blob3['labels'].values, sort=False))

#PLotting the imbalanced dataset
sn.FacetGrid(data=df_blob3,hue='labels',size=3).map(plt.scatter,'feature_1','feature_2')
plt.legend()
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()


'''
Technique 4:Creating synthetic features (ADASYN)
Perform over-sampling using Adaptive Synthetic (ADASYN) sampling approach for imbalanced datasets
'''
from imblearn.over_sampling import ADASYN
rus4 = ADASYN(ratio='auto',n_neighbors=3,random_state=42)
X4_res, y4_res = rus4.fit_sample(X_res, y_res)
y4_res=y4_res.reshape(20000,1)
dataset_blob4= np.concatenate((X4_res,y4_res),axis=1)
df_blob4=pd.DataFrame(dataset_blob4,columns=('feature_1','feature_2','labels'))

print('The number of samples of class 0 and 1:',pd.value_counts(df_blob4['labels'].values, sort=False))

#PLotting the imbalanced dataset
sn.FacetGrid(data=df_blob4,hue='labels',size=3).map(plt.scatter,'feature_1','feature_2')
plt.legend()
plt.xlabel('feature_1')
plt.ylabel('feature_2')
plt.show()

'''
There are multiple Ensemble Methods for handling imbalanced datasets too
'''