from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

'''
This function takes in the training data and performs imputation and preprocessing.
'''
def run_models_with_imputation(X, y):
    # Adjust the labels for compatibility with XGBoost
    y_adjusted = y - 1
    
    # Initialize models
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)
    ab_model = AdaBoostClassifier(random_state=42)
    
    # Create a scorer
    scorer = make_scorer(f1_score, pos_label=1)
    
    # Perform 5-fold cross-validation and print average F1 score
    for model, name in zip([xgb_model, rf_model, gb_model, ab_model], 
                           ['XGBoost', 'Random Forest', 'Gradient Boosting', 'AdaBoost']):
        if name == 'XGBoost':
            f1_scores = cross_val_score(model, X, y_adjusted, cv=5, scoring=scorer)
        else:
            f1_scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
        print(f"Average F1 Score for {name}: {f1_scores.mean()}")



'''
This function takes in the training data, categorical and numerical columns, and an imputer and returns the imputed and preprocessed data.
'''
def impute_and_preprocess(data, categorical_cols, numerical_cols, imputer):
    data_imputed = data.copy()
    data_imputed[numerical_cols] = imputer.fit_transform(data_imputed[numerical_cols])
    for col in categorical_cols:
        mode_val = data[col].mode()[0]
        data_imputed[col].fillna(mode_val, inplace=True)
        le = LabelEncoder()
        data_imputed[col] = le.fit_transform(data_imputed[col])
    return data_imputed

# Read in the training data
train_data = pd.read_csv('data_training.csv')

# Identified categorical and numerical columns
categorical_cols = ['Feature_2', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7']
numerical_cols = ['Feature_1', 'Feature_3', 'Feature_8', 'Feature_9', 'Feature_10', 'Feature_11', 'Feature_12', 'Feature_13', 'Feature_14', 'Feature_15', 'Feature_16', 'Feature_17', 'Feature_18', 'Feature_19']

# Create different imputers
from sklearn.impute import SimpleImputer, KNNImputer
mean_imputer = SimpleImputer(strategy='mean')
mode_imputer = SimpleImputer(strategy='most_frequent')
knn_imputer = KNNImputer(n_neighbors=2)

# Loop through each imputer
for imputer, name in zip([mean_imputer, mode_imputer, knn_imputer], ['Mean', 'Mode', 'KNN']):
    print(f"Results for {name} imputation:")
    train_data_imputed = impute_and_preprocess(train_data, categorical_cols, numerical_cols, imputer)
    X = train_data_imputed.drop('Label', axis=1)
    y = train_data_imputed['Label']
    run_models_with_imputation(X, y)

# just for dropped rows
def drop_rows_and_preprocess(data, categorical_cols, numerical_cols):
    data_dropped = data.dropna().copy()
    for col in categorical_cols:
        le = LabelEncoder()
        data_dropped[col] = le.fit_transform(data_dropped[col])
    return data_dropped

# Drop rows with missing values and preprocess
train_data_dropped = drop_rows_and_preprocess(train_data, categorical_cols, numerical_cols)
X_dropped = train_data_dropped.drop('Label', axis=1)
y_dropped = train_data_dropped['Label']

# Run models with dropped rows preprocessing
print("Results for dropping rows with missing values:")
run_models_with_imputation(X_dropped, y_dropped)
