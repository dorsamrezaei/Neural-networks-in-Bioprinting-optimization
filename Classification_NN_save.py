import pandas as pd
import numpy as np
from numpy import mean
import matplotlib as mp
import matplotlib.pyplot as plt
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_validate, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDRegressor, SGDClassifier 
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE #Recursive Feature Elimination (RFE)_method of feature selection
from keras import regularizers
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import make_scorer, mean_squared_error
from process_data import *
from sys import exit
from keras.optimizers import Adam

    
bioprint_df = pd.read_excel('Original_dataset.xlsx') 
bioprint_df = bioprint_df.drop(['Reference'], axis = 1)
bioprint_df = bioprint_df.drop(['DOI'], axis = 1)
#To  return the total number of NaN values in the entire DataFrame:
bioprint_df.isna().sum()
# Replacthe different lables of endothelials cells to one lable
bioprint_df['Cell type'].replace('Endothelial cells ', 'Endothelial cells', inplace=True)
bioprint_df['Cell type'].replace('Endothelial cells \x02', 'Endothelial cells', inplace=True)
bioprint_df['Cell type'].replace('embryonic stem cells ', 'Embryonic stem cells ', inplace=True)
bioprint_df['Cell type'].replace('Endothelial cells _x0002_', 'Endothelial cells', inplace=True)
# Remove rows where Alginate and Gelatin both are 0
bioprint_df = bioprint_df[~((bioprint_df['Alginate_Concentration(%w/v)'] == 0) & (bioprint_df['Gelatin_Concentration(%w/v)'] == 0))] #One-Hot Encoding
bioprint_df = pd.get_dummies(bioprint_df, columns=['Cell type'])   
#Imputing with KNN
imputer_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent') #imputing mode value into missing values for temperatures
bioprint_df.loc[:,['Cartridge_Temperature(°C)','Bed_Temperature(°C)']] = imputer_mode.fit_transform(bioprint_df.loc[:,['Cartridge_Temperature(°C)','Bed_Temperature(°C)']])
#Drop instances without cell viability values
bioprint_df = bioprint_df[bioprint_df['Viability(%)'].notna()]
#Drop nonprinting instances (instances were extrusion pressure is zero)
bioprint_df = bioprint_df.drop(bioprint_df[bioprint_df['Printing_Pressure(kPa)'] == 0 ].index)
#Imputing Values
bioprint_df.isna().sum() #produces a list of each variable’s number of null values
#Imputation of numerical/continuous values databases
imputer_knn = KNNImputer(n_neighbors = 10, weights = "uniform") #imputing mode value into missing values
bioprint_df.iloc[:,1:44] = imputer_knn.fit_transform(bioprint_df.iloc[:,1:44]) #used for cell viability dataset preprocessing    
#randomly shuffle all the rows of the DataFrame and return them as a shuffled sample:
# Set the random seed
np.random.seed(42)
# Shuffle the DataFrame
bioprint_df = bioprint_df.sample(frac=1).reset_index(drop=True)
# Convert the target variable to binary
y = (bioprint_df['Viability(%)'].values > 70).astype(int) # used to be 70 (ask why)
x = bioprint_df.drop('Viability(%)', axis=1)
x, x_val, y, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
optimizer = Adam(learning_rate=0.01)
# Define the model architecture
def estimator():
    model = keras.Sequential()
    model.add(keras.layers.Dense(44, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.05))) 
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', metrics=["accuracy"], optimizer=optimizer)
    return model

# Define the number of folds for cross-validation
num_folds = 10
# Initialize lists to store training and test accuracy across folds
train_acc_list = []
train_precision_list = []
train_recall_list = []
test_acc_list = []
test_precision_list = []
test_recall_list = []
val_acc_list = []
val_precision_list = []
val_recall_list = []
models = []

# Create the KFold object
kf = KFold(n_splits=num_folds, shuffle=True)
x_dataframe=x.copy(deep=True)
# Loop over the folds
for idx,(train_index, test_index) in enumerate(kf.split(x)):   
   # Split the data into training and test sets for this fold
    x_fold_train=x.iloc[train_index]
    y_fold_train = y[train_index]
    x_fold_test=x.iloc[test_index]
    y_fold_test = y[test_index]
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    x_fold_train = scaler.fit_transform(x_fold_train)
    x_fold_test = scaler.transform(x_fold_test)
    model = KerasClassifier(build_fn=estimator, epochs=1000, batch_size=16, verbose=1)                                                                                                          
    history=model.fit(x_fold_train, y_fold_train, verbose=1) 
    # Evaluate the model on the training set for this fold
    train_acc = model.score(x_fold_train, y_fold_train)
    train_acc_list.append(train_acc)
    y_fold_train_pred = model.predict(x_fold_train)
    train_precision = precision_score(y_fold_train, y_fold_train_pred.round())
    train_precision_list.append(train_precision)
    train_recall = recall_score(y_fold_train, y_fold_train_pred.round())
    train_recall_list.append(train_recall)
    print(f"Train acc: {train_acc} fold {idx}")
    print(f"Train precision: {train_precision} fold {idx}")
    print(f"Train recall: {train_recall} fold {idx}")
    # Evaluate the model on the test set for this fold
    test_acc = model.score(x_fold_test, y_fold_test)
    test_acc_list.append(test_acc)
    y_fold_test_pred = model.predict(x_fold_test)
    test_precision = precision_score(y_fold_test, y_fold_test_pred.round())
    test_precision_list.append(test_precision)
    test_recall = recall_score(y_fold_test, y_fold_test_pred.round())
    test_recall_list.append(test_recall)
    print(f"Test acc: {test_acc} fold {idx}")
    print(f"Test_precision: {test_precision} fold {idx}")
    print(f"Test recall: {test_recall} fold {idx}")
    # Save the trained model for this fold
    models.append(model)
    
best_model_index = np.argmax(test_acc_list)
best_model = models[best_model_index]   
# Compute the average training and test accuracy across all folds
avg_train_acc = sum(train_acc_list) / num_folds
avg_test_acc = sum(test_acc_list) / num_folds
# Compute the average training and test accuracy across all folds
avg_train_acc = sum(train_acc_list) / num_folds
avg_train_precision = sum(train_precision_list) / num_folds
avg_train_recall = sum(train_recall_list) / num_folds
avg_test_acc = sum(test_acc_list) / num_folds
avg_test_precision = sum(test_precision_list) / num_folds
avg_test_recall = sum(test_recall_list) / num_folds
# Print the results
print(f'Average training set accuracy: {avg_train_acc}')
print(f'Average training set precision: {avg_train_precision}')
print(f'Average training set recall: {avg_train_recall}')
print(f'Average test set accuracy: {avg_test_acc}')
print(f'Average test set precision: {avg_test_precision}')
print(f'Average test set recall: {avg_test_recall}')

acc_mean = avg_test_acc
acc_std = np.std(test_acc_list)
precision_mean = avg_test_precision
precision_std = np.std(test_precision_list)
recall_mean = avg_test_recall
recall_std = np.std(test_recall_list)
acc_mean = avg_train_acc
acc_std = np.std(train_acc_list)
precision_mean = avg_train_precision
precision_std = np.std(train_precision_list)
recall_mean = avg_train_recall
recall_std = np.std(train_recall_list)
#val set evaluation, not seen by model
x = scaler.fit_transform(x,y)
x_val = scaler.transform(x_val)
y_val = np.ravel(y_val)
val_acc = model.score(x_val, y_val)
y_val_pred = model.predict(x_val)
val_precision = precision_score(y_val, y_val_pred.round())
val_recall = recall_score(y_val, y_val_pred.round())

print(f'Validation set accuracy: {val_acc}')
print(f"Validation set precision: {val_precision}")
print(f"Validation set recall: {val_recall}")

# model.save('my_model.h5')
model.model.save('saved_classification_NN_1.h5')
# Set the plot parameters
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(3)
width = 0.35
# Define the data and standard deviation values for the training set
train_data = [acc_mean, precision_mean, recall_mean]
train_std = [acc_std,precision_std,recall_std]
# Define the data and standard deviation values for the test set
test_data = [val_acc, val_precision, val_recall]
test_std = [0,0,0]
# Plot the performance metrics for the training set with error bars
ax.bar(x_pos - width/2, train_data, width, align='center', alpha=0.5, color='darkgreen', label='Train Set', yerr=train_std, capsize=10)
# Plot the performance metrics for the test set with error bars
ax.bar(x_pos + width/2, test_data, width, align='center', alpha=0.5, color='darkblue', label='Test Set', capsize=10)
# Set the plot labels and title
ax.set_xticks(x_pos)
ax.legend()
plt.rcParams.update({'font.size': 22, 'font.family': 'Times New Roman'})
plt.show()
