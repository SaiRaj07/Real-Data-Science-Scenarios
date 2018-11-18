######################################## TASK DESCRIPTION #######################################
'''
Handling Missing Value (Different Strategies)
Feature Details
0. Number of times pregnant.
1. Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
2. Diastolic blood pressure (mm Hg).
3. Triceps skinfold thickness (mm).
4. 2-Hour serum insulin (mu U/ml).
5. Body mass index (weight in kg/(height in m)^2).
6. Diabetes pedigree function.
7. Age (years).
8. Class variable (0 or 1)
'''

####################################### Importing the library and Dataset #######################
import pandas as pd
import numpy as np

#Importing the datasets 
dataset_diabetes =pd.read_csv('diabetes1.csv')
dataset_diabetes.columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
dataset_diabetes.shape

dataset_diabetes.head()
dataset_diabetes.tail()

#Detecting the missing values
#Few simple stats on the data
dataset_diabetes.describe()

#Lets detect number of samples with missing values
#Zero values indicate missing values
(dataset_diabetes[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] == 0).sum()

#Lets replace missing values with Nan
cols =['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
dataset_diabetes[cols] = dataset_diabetes[cols].replace(0,np.NaN)

dataset_diabetes.head()
dataset_diabetes.tail()

############################# Methods of Handling the Missing data ##############################
'''
1) Removing rows with missing values
2) General Imputation
3) Class/label based Imputation
4) Model based Imputation
5) Creating binary missing features
'''

############################## Remove the rows with missing data ################################
dataset_removed_samples=dataset_diabetes.dropna()
dataset_removed_samples.shape
dataset_removed_samples.head()

'''
We can see the size of the dataset has reduced drastically if we remove the rows with 
missing values Not the best technique if we are limited with the amount of data we have
'''

##################################### GENERAL IMPUTATION ########################################
#Technique 2 
#General Imputation  (Mean, Median,Mode)
from sklearn.preprocessing import Imputer #Handling NAN Values
mean_imputer = Imputer(missing_values='NaN',strategy='mean').fit_transform(dataset_diabetes)
mean_imputer.shape

#More robust to the outlier
from sklearn.preprocessing import Imputer #Handling NAN Values
median_imputer = Imputer(missing_values='NaN',strategy='median').fit_transform(dataset_diabetes)
median_imputer.shape

from sklearn.preprocessing import Imputer #Handling NAN Values
mode_imputer = Imputer(missing_values='NaN',strategy='most_frequent').fit_transform(dataset_diabetes)
mode_imputer.shape

'''
Not the best strategy to use general imputation since mean , median ,mode can be different
with respect to different target lables for classification task
'''
###################################### LABEL IMPUTATION ########################################

#Technique 3
#Class Based Imputation
dataset_diabetes_groupby=dataset_diabetes.groupby(by='Outcome')

dataset_diabetes_groupby.aggregate(np.mean)

group_zero=dataset_diabetes_groupby.get_group(0)
group_one=dataset_diabetes_groupby.get_group(1)

from sklearn.preprocessing import Imputer #Handling NAN Values
mean_class_zero_imputer = Imputer(missing_values='NaN',strategy='mean').fit_transform(group_zero)
mean_class_zero_imputer.shape

from sklearn.preprocessing import Imputer #Handling NAN Values
median_class_zero_imputer = Imputer(missing_values='NaN',strategy='median').fit_transform(group_zero)
median_class_zero_imputer.shape

from sklearn.preprocessing import Imputer #Handling NAN Values
mean_class_one_imputer = Imputer(missing_values='NaN',strategy='mean').fit_transform(group_one)
mean_class_one_imputer.shape

from sklearn.preprocessing import Imputer #Handling NAN Values
median_class_one_imputer = Imputer(missing_values='NaN',strategy='median').fit_transform(group_one)
median_class_one_imputer.shape

#Median Class label Imputed dataset
median_imputed_dataset=np.concatenate((median_class_zero_imputer,median_class_one_imputer))
np.random.shuffle(median_imputed_dataset)

#Mean Class label Imputed dataset
mean_imputed_dataset=np.concatenate((mean_class_zero_imputer,mean_class_one_imputer))
np.random.shuffle(mean_imputed_dataset)

'''
Better techniques as compared to previous two techniques
'''
###################################### KNN Model Based Imputation ###################################
'''
1)Lets build a KNN based regression  approach to predict the missing values for insulin.
2)We can do this easily using fancyimpute KNN.
3)Nearest neighbor imputations which weights samples using the mean squared difference on 
 features for which two rows both have observed data
'''
from fancyimpute import KNN,IterativeImputer
X_filled_knn = KNN(k=3).fit_transform(dataset_diabetes)

######################################### Iterative Imputation ##################################
'''
Advanced Imputation Strategy:
A strategy for imputing missing values by modeling each feature with missing values as a function 
of other features in a round-robin fashion
'''
X_filled_ii = IterativeImputer().fit_transform(dataset_diabetes)

############## Creating binary missing features (missing values-source of information) ##########
'''
Sometimes in few cases, missing values can be a source of information or may be in someway 
indicating some pattern.
Thus we create binary missing features and concatenate them with any of
1) General Imputation
2) Label Imputation
3) KNN Imputatiom
4) Iterative Imputation    
'''
#Generating binary features
X = dataset_diabetes.as_matrix()
a=[]
for x in X:
    for y in x:
        if y>=0:
           a.append(0)
        else:
            a.append(1)
b=np.asarray(a)
b=b.reshape(768,9)
    
# New features representing the missing information with General Imputation
Binary_features_miss=np.concatenate((median_imputer,b),axis=1)
