import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from numpy import mean
import matplotlib as mp
import matplotlib.pyplot as plt
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, MaxAbsScaler
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
from sklearn.metrics import r2_score
#from process_data import *
from sys import exit
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import make_scorer, mean_squared_error


def process_data():
    
    bioprint_df = pd.read_excel('Original_dataset_1.xlsx') 

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
    bioprint_df = bioprint_df[~((bioprint_df['Alginate_Concentration(%w/v)'] == 0) & (bioprint_df['Gelatin_Concentration(%w/v)'] == 0))] 


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
    
    
    y = bioprint_df['Viability(%)'].values
    x = bioprint_df.drop('Viability(%)', axis = 1)
    x_original=x.copy(deep=True)
    #One-Hot Encoding
    #x = pd.get_dummies(x, columns=['Cell type'])
    
    #Label Encoding of cell type
    #x['Cell type'] = x['Cell type'].astype('category').cat.codes
   

    x.to_csv (r'processed_x_shuffled.csv',header=True)
    # y.to_csv (r'y.csv',header=True)
    return x,y,x_original

from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.01)
# Define the model architecture
# Define the function to create the model
def estimator():
    model = keras.Sequential()
    model.add(keras.layers.Dense(44, activation='tanh'))
   # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(120, activation='tanh', kernel_regularizer=regularizers.l2(0.01))) # used to be 0.05
    #model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='linear'))
    # Compile the model
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Read and preprocess the data
x, y,z = process_data()

y = MinMaxScaler().fit_transform(y.reshape(-1, 1))


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

# Create the pipeline

# Define the preprocessing steps
# Define the preprocessing steps
categorical_cols = ["Cell type"]
numeric_cols = [col for col in X_train.columns if "Cell type" not in col]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_cols),
        ("num", MinMaxScaler(), numeric_cols)
    ])

# Define the pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimator", KerasRegressor(build_fn=estimator, epochs=1000, batch_size=5, verbose=0))
])
# Fit the pipeline
pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Compute MSE and R2 on the validation set
y_val_pred = pipeline.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

# Print the results
print(f"Training set MSE: {mse_train:.2f}")
print(f"Training set R2: {r2_train:.2f}")
print(f"Validation set MSE: {mse_val:.2f}")
print(f"Validation set R2: {r2_val:.2f}")

# Compute feature importances
result = permutation_importance(pipeline, x, y, n_repeats=10, random_state=0)

# Plot the 10 most important features
sorted_idx = result.importances_mean.argsort()[-10:]
fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=x.columns[sorted_idx])
ax.set_title("Permutation Importances (Top 10)")
fig.tight_layout()
plt.show()

