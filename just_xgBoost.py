'''
Since XGBoost has the lowest MSE score, but did not compute an F1 score, 
this file is solely dedicated to XGBoost.
'''

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Read in the training data
train_data = pd.read_csv('data_training.csv')
# Read in the test data
test_data = pd.read_csv('data_test.csv')

# Identified categorical and numerical columns
categorical_cols = ['Feature_2', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7']
numerical_cols = ['Feature_1', 'Feature_3', 'Feature_8', 'Feature_9', 'Feature_10', 'Feature_11', 'Feature_12', 'Feature_13', 'Feature_14', 'Feature_15', 'Feature_16', 'Feature_17', 'Feature_18', 'Feature_19']

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

# Assuming your preprocessed feature matrix is X_mode and labels are y_mode
# Adjust the labels for compatibility with XGBoost
y_mode_adjusted = y_mode - 1

# Initialize XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Create a scorer
scorer = make_scorer(f1_score, pos_label=1)  # Note the pos_label is adjusted to 1

# Perform 5-fold cross-validation
f1_scores_xgb_adjusted = cross_val_score(xgb_model, X_mode, y_mode_adjusted, cv=5, scoring=scorer)

# Calculate average F1 score
average_f1_score_xgb_adjusted = np.mean(f1_scores_xgb_adjusted)
print("Average F1 Score for XGBoost:", average_f1_score_xgb_adjusted)

# Average F1 Score for XGBoost: 0.9062059272216227