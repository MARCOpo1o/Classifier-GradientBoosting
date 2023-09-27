'''
 use the training set to build your model and then apply your model to make predictions for the samples in 
 the test set. Save your predictions in a file "P1_test_output.csv", in which each row is the prediction 
 of the corresponding row in the test set. Please do not add a header row in "P1_test_output.csv".
 I will be using Gradient Boosting Classifier for this problem. and preprocess the data using mode imputation.
'''
import numpy as np 
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Read in the training data
train_data = pd.read_csv('data_training.csv')
# Read in the test data
test_data = pd.read_csv('data_test.csv')

# Identified categorical and numerical columns
categorical_cols = ['Feature_2', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7']
numerical_cols = ['Feature_1', 'Feature_3', 'Feature_8', 'Feature_9', 'Feature_10', 'Feature_11',
                  'Feature_12', 'Feature_13', 'Feature_14', 'Feature_15', 'Feature_16', 'Feature_17', 'Feature_18',
                  'Feature_19']

# Function to preprocess data using mode imputation
def mode_impute_and_preprocess(data, categorical_cols, numerical_cols):
    data_imputed = data.copy()
    for col in numerical_cols:
        mode_val = data[col].mode()[0]
        data_imputed[col].fillna(mode_val, inplace=True)
    for col in categorical_cols:
        mode_val = data[col].mode()[0]
        data_imputed[col].fillna(mode_val, inplace=True)
        le = LabelEncoder()
        data_imputed[col] = le.fit_transform(data_imputed[col])
    return data_imputed

# Preprocess data using mode imputation
train_data_mode = mode_impute_and_preprocess(train_data, categorical_cols, numerical_cols)
X_mode = train_data_mode.drop('Label', axis=1)
y_mode = train_data['Label']

# Initialize Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)

#use the training set to build your model and then apply your model to make predictions for the samples in the test set. Save your predictions in a file "P1_test_output.csv", in which each row is the prediction of the corresponding row in the test set. Please do not add a header row in "P1_test_output.csv"
gb_model.fit(X_mode, y_mode)
test_data_mode = mode_impute_and_preprocess(test_data, categorical_cols, numerical_cols)
X_test_mode = test_data_mode
y_test_mode = gb_model.predict(X_test_mode)
pd.DataFrame(y_test_mode).astype(int).to_csv("P1_test_output.csv", header=False, index=False)


