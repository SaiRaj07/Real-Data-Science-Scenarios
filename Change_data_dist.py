'''
Detecting the change in data distribution during training and testing period
Co-variant shift = Shift in the independent variables with time
Prior probability shift= Shift in the target variable
Concept Shift = Shift relationship between independent variable and dependent variable
'''

################################### Importing the libraries #######################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split

################################ Importing the dataset #############################################
train = pd.read_csv('C:\\Users\\Sairaj\\Downloads\\all\\application_train.csv')
test = pd.read_csv('C:\\Users\\Sairaj\\Downloads\\all\\application_test.csv')

########################################### Imputation ###########################################
'''
DataFrameImputer function is taken from stackoverflow with few modifications
'''
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with median of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
############################################## Imputed Data ###################################################
train = DataFrameImputer().fit_transform(train)
test = DataFrameImputer().fit_transform(test)

########################################### Label Encoding ###############################################
for col in train.columns:
    if train[col].dtype == 'object':
      train[col] = train[col].astype('category')
      train[col] = train[col].cat.codes

for col in test.columns:
    if test[col].dtype == 'object':
      test[col] = test[col].astype('category')
      test[col] = test[col].cat.codes
      
######################################## Data Prep for Feature importance ###############################################
ran_train=train
y_ran=ran_train['TARGET']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(ran_train, y_ran, test_size=0.33, random_state=42)
    
################################################################################################
train = train.drop('TARGET',axis=1) #dropping the target class
train['set'] = 0 #New target for train
test['set'] = 1 #New target for test

#Creating a test and train dataset with of 5000 samples each
train_df = train.sample(5000, random_state=344)
test_df = test.sample(5000, random_state=433)

new_data = train_df.append(test_df)
y_label = new_data['set']
new_data = new_data.drop('set',axis=1)

Xx_train, Xx_test, yx_train, yx_test = train_test_split(new_data, y_label, test_size=0.33, random_state=42)
    
########################################## Detecting Datashift #################################################
random_forest =RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
random_forest.fit(Xx_train, yx_train)
pred = random_forest.predict(Xx_test)
a=accuracy_score( yx_test, pred,normalize=True)* float(100)
conf_mat = confusion_matrix(yx_test, pred)

'''
With high accuracy classifies whether data comes from training set or testing set indicating
that train and test set have different distribution
'''
'''
The change in distribution takes place if feature behaviour changes with time, this is 
called as unstable features  
'''
######################################## Random Forest #########################################################
'''
Detecting unstable features causing change in distribution.
If the value of AUC-ROC for a particular feature is greater than 0.70,
we classify that feature as drifting.
'''
random_model = RandomForestClassifier(n_estimators = 30, max_depth = 5, min_samples_leaf = 7)
drop_features= []
for col in new_data.columns:
    score = cross_val_score(random_model,pd.DataFrame(new_data[col]),y_label,cv=2,scoring='roc_auc')
    if np.mean(score) > 0.7:
        drop_features.append(col)
    print(col,np.mean(score))

##################################### Detecting important features ##############################
'''
Checking feature importance before dropping
Split train data into train and test data
'''

random_forest_detect =RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
random_forest_detect.fit(Xr_train, yr_train)
pred_detect = random_forest_detect.predict(Xr_test)
a_detect=accuracy_score( yr_test, pred_detect,normalize=True)* float(100)
conf_mat_detect = confusion_matrix(yr_test, pred_detect)
imp = random_forest.feature_importances_
indices = np.argsort(imp)[::-1][:20]
features=train.columns.values

#Plotting the feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.bar(range(len(indices)), imp[indices], color = 'b', align='center')
plt.xticks(range(len(indices)),features[indices], rotation='vertical')
plt.xlim([-1,len(indices)])
plt.show()
################################### Finding Common Features #####################################
'''
Find features common in drop feature list and important feature list and removes the rest and
build the model
''' 
