import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifier

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

# Initialize models
decision_tree = DecisionTreeClassifier(random_state=42)
# Initialize models with updated parameters
logistic_reg = LogisticRegression(random_state=42, max_iter=3000)  # Increased max_iter
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

knn = KNeighborsClassifier()
naive_bayes = GaussianNB()
svm_rbf = SVC(kernel='rbf', random_state=42)
mlp = MLPClassifier(random_state=42)

ada_boost = AdaBoostClassifier(random_state=42)
qda = QuadraticDiscriminantAnalysis()

random_forest = RandomForestClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)
bagging_tree = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)
ridge_classifier = RidgeClassifier(random_state=42)




# List of models to evaluate
new_models = [decision_tree, logistic_reg, knn, naive_bayes, svm_rbf, mlp, xgb, ada_boost, qda, random_forest, gradient_boosting, bagging_tree, ridge_classifier]
new_model_names = ['Decision Trees', 'Logistic Regression', 'k-NN', 'Naive Bayes', 'SVM-RBF', 'Neural Networks', 'XGBoost', 'AdaBoost', 'QDA', 'Random Forest', 'Gradient Boosting', 'Bagging', 'Ridge Classifier']

# Scoring metric
scorer = make_scorer(f1_score, pos_label=2)

# List to store average F1 scores for each model
new_average_f1_scores = {}

# List to store average MSE scores for each model
new_average_mse_scores = {}

# Re-map labels for XGBoost
y_mode_remap = y_mode - 1  # Maps [1, 2] to [0, 1]

# Perform 5-fold cross-validation and store the F1 scores and MSE scores
for model, name in zip(new_models, new_model_names):
    target = y_mode_remap if name == 'XGBoost' else y_mode  # Use remapped labels for XGBoost
    try:
        f1_scores = cross_val_score(model, X_mode, target, cv=5, scoring=scorer)
        mse_score = cross_val_score(model, X_mode, target, cv=5, scoring='neg_mean_squared_error')
        new_average_f1_scores[name] = np.mean(f1_scores)
        new_average_mse_scores[name] = np.mean(mse_score)
    except Exception as e:
        print(f"Error in model {name}: {e}")
        new_average_f1_scores[name] = 'Error'

print(new_average_f1_scores)
print(new_average_mse_scores) #  shows that XGBoost has the lowest MSE score
'''
{'Decision Trees': 0.811615573745736, 
'Logistic Regression': 0.6102598274535913, 
'k-NN': 0.6292970132676996, 
'Naive Bayes': 0.6473989565718904, 
'SVM-RBF': 0.0, 
'Neural Networks': 0.43195726295515147, 
'XGBoost': nan, 
'AdaBoost': 0.862096399063032, 
'QDA': 0.6455337976232667, 
'Random Forest': 0.8703740067835521, 
'Gradient Boosting': 0.8895496960256934, 
'Bagging': 0.8591120317627526, 
'Ridge Classifier': 0.598582012704433}
'''
