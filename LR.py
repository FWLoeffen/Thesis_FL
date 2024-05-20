from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# Setting a global random seed
np.random.seed(42)
random.seed(42)

# Load the w2v dataset
file_path = 'df_w2v_skimmed.csv'
data = pd.read_csv(file_path)

# Define the target and features
X = data.drop(['Cause_of_disengagement', 'Word2Vec_Clusters', 'Word2Vec_Cluster_Labels'], axis=1)
y = data['Word2Vec_Clusters']

# Set up Stratified K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'roc_auc': []
}

# Training and evaluating the model using cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics for the current fold
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
    # Handle multi-class AUC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    metrics['roc_auc'].append(roc_auc_score(y_test_binarized, y_prob, multi_class='ovr'))

# Calculate and print average of metrics
for metric_name, values in metrics.items():
    
    print(f"Average {metric_name.capitalize()}: {np.mean(values)}")

# W2Vset per class

from sklearn.metrics import classification_report, confusion_matrix


# Generate a classification report
class_report = classification_report(y_test, y_pred, target_names=np.unique(y).astype(str), output_dict=True)

# Print detailed classification report
print("Detailed classification report:")
for label, metrics in class_report.items():
    if label.isdigit():  # Check if the label represents a class
        print(f"Metrics for class {label}:")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  F1-score: {metrics['f1-score']:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

#Compute ROC-AUC per class if applicable
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Binarize the output
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
roc_auc = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr')
print(f"ROC-AUC (One-vs-Rest): {roc_auc:.2f}")

# Randomized search w2v

# Define the parameter grid for randomized search
param_grid = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'l1_ratio': np.linspace(0, 1, 10)  # Only used for 'elasticnet'
}

# Initialize the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Initialize the RandomizedSearchCV
lr_random = RandomizedSearchCV(estimator=lr_model, param_distributions=param_grid,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Perform the randomized search
lr_random.fit(X, y)

# Print the best parameters found by the search
print(f"Best Parameters: {lr_random.best_params_}")

# Use the best estimator for further evaluation
best_lr_model = lr_random.best_estimator_

# Training and evaluating the best model using cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the best model on the training set
    best_lr_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = best_lr_model.predict(X_test)
    y_prob = best_lr_model.predict_proba(X_test)
    
    # Calculate metrics for the current fold
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
    # Handle multi-class AUC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    metrics['roc_auc'].append(roc_auc_score(y_test_binarized, y_prob, multi_class='ovr'))

# Calculate and print average of metrics
for metric_name, values in metrics.items():
    print(f"Average {metric_name.capitalize()}: {np.mean(values)}")

### Now TF-IDF

# Load the tf-idf dataset
file_path = 'df_tfidf_skimmed.csv'
data = pd.read_csv(file_path)


# Define the target and features
X = data.drop(['Cause_of_disengagement', 'TFIDF_Clusters', 'TFIDF_Cluster_Labels'], axis=1)
y = data['TFIDF_Clusters']

# Set up Stratified K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'roc_auc': []
}

# Training and evaluating the model using cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics for the current fold
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
    # Handle multi-class AUC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    metrics['roc_auc'].append(roc_auc_score(y_test_binarized, y_prob, multi_class='ovr'))

# Calculate and print average of metrics
for metric_name, values in metrics.items():
    print(f"{metric_name.capitalize()} per fold: {values}")
    print(f"Average {metric_name.capitalize()}: {np.mean(values)}")

    
# tfidf set per class

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test) 

# Generate a classification report
class_report = classification_report(y_test, y_pred, target_names=np.unique(y).astype(str), output_dict=True)

# Print detailed classification report
print("Detailed classification report:")
for label, metrics in class_report.items():
    if label.isdigit():  # Check if the label represents a class
        print(f"Metrics for class {label}:")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  F1-score: {metrics['f1-score']:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Binarize the output
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
roc_auc = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr')
print(f"ROC-AUC (One-vs-Rest): {roc_auc:.2f}")

##With oversampling
# Define the target and features
X = data.drop(['Cause_of_disengagement', 'Word2Vec_Clusters', 'Word2Vec_Cluster_Labels'], axis=1)
y = data['Word2Vec_Clusters']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'roc_auc': []
}

# Initialize the RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Training and evaluating the model using cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply oversampling
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics for the current fold
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
    # Handle multi-class AUC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    metrics['roc_auc'].append(roc_auc_score(y_test_binarized, y_prob, multi_class='ovr'))

# Calculate and print average of metrics
for metric_name, values in metrics.items():
    print(f"Average {metric_name.capitalize()}: {np.mean(values)}")

# Load the tfidf dataset
file_path = 'df_tfidf_skimmed.csv'
data = pd.read_csv(file_path)

# Define the target and features
X = data.drop(['Cause_of_disengagement', 'TFIDF_Clusters', 'TFIDF_Cluster_Labels'], axis=1)
y = data['TFIDF_Clusters']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'roc_auc': []
}

# Initialize the RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Training and evaluating the model using cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply oversampling
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics for the current fold
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))
    # Handle multi-class AUC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
    metrics['roc_auc'].append(roc_auc_score(y_test_binarized, y_prob, multi_class='ovr'))

# Calculate and print average of metrics
for metric_name, values in metrics.items():
    print(f"Average {metric_name.capitalize()}: {np.mean(values)}")

