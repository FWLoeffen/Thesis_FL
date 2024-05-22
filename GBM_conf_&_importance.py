from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score


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

# Arrays to store true values and predicted probabilities for ROC AUC and PR curves
y_true_all = np.array([])
y_prob_all = np.empty((0, len(np.unique(y))))

# Training and evaluating the model using cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train the Gradient Boosting classifier
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
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
    
    # Store the true values and predicted probabilities for ROC and PR curves
    y_true_all = np.append(y_true_all, y_test)
    y_prob_all = np.vstack((y_prob_all, y_prob))

# Calculate and print average of metrics
for metric_name, values in metrics.items():
    print(f"Average {metric_name.capitalize()}: {np.mean(values)}")

# Binarize the true values for multiclass
y_true_binarized = label_binarize(y_true_all, classes=np.unique(y))

# Compute ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_true_binarized.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_prob_all[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_prob_all.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute Precision-Recall curve and average precision for each class
precision = dict()
recall = dict()
average_precision = dict()

for i in range(y_true_binarized.shape[1]):
    precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_prob_all[:, i])
    average_precision[i] = average_precision_score(y_true_binarized[:, i], y_prob_all[:, i])

# Compute micro-average Precision-Recall curve and average precision
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_binarized.ravel(), y_prob_all.ravel())
average_precision["micro"] = average_precision_score(y_true_binarized, y_prob_all, average="micro")

# Plot ROC AUC Curve
plt.figure(figsize=(14, 8))
plt.plot(fpr["micro"], tpr["micro"], label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

for i in range(y_true_binarized.shape[1]):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('ROC AUC Curve for Gradient Boosting Classifier', fontsize=17)
plt.legend(loc="lower right", fontsize=12)
plt.tight_layout()
plt.show()

# Plot Precision-Recall Curve
plt.figure(figsize=(14, 8))
plt.plot(recall["micro"], precision["micro"], label='Micro-average Precision-Recall curve (area = {0:0.2f})'.format(average_precision["micro"]))

for i in range(y_true_binarized.shape[1]):
    plt.plot(recall[i], precision[i], label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.title('Precision-Recall Curve for Gradient Boosting Classifier', fontsize=17)
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# CONFUSION MATRIX!

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate predicted class labels from predicted probabilities
y_pred_all = np.argmax(y_prob_all, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_true_all, y_pred_all)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix for Gradient Boosting Classifier', fontsize=16)
plt.show()

# Generate predicted class labels from predicted probabilities
y_pred_all = np.argmax(y_prob_all, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_true_all, y_pred_all)

# Normalize the confusion matrix by row 
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
plt.xlabel('Predicted Class', fontsize=14)
plt.ylabel('True Class', fontsize=14)
plt.title('Confusion Matrix for Gradient Boosting Classifier', fontsize=14)
plt.show()

# Extract feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for better visualization
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by feature importance in descending order
features_df = features_df.sort_values(by='Importance', ascending=False)

# Generate a color palette
colors = plt.cm.viridis(np.linspace(0, 1, len(features_df)))

# Plot the feature importances with multiple colors
plt.figure(figsize=(12, 8))
bars = plt.barh(features_df['Feature'], features_df['Importance'], color=colors)
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance for Gradient Boosting Classifier', fontsize=16)
plt.gca().invert_yaxis()  # To display the most important features on top
plt.show()

# Filter features for 'manufacturer_'
manufacturer_features_df = features_df[features_df['Feature'].str.startswith('Manufacturer_')]

# Plot for 'manufacturer_' features
plt.figure(figsize=(12, 8))
bars = plt.barh(manufacturer_features_df['Feature'], manufacturer_features_df['Importance'], color=colors[:len(manufacturer_features_df)])
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Manufacturer Feature', fontsize=14)
plt.title('Feature Importance for Manufacturer Features', fontsize=16)
plt.gca().invert_yaxis()  # To display the most important features on top
plt.show()

# Filter features for 'location_'
location_features_df = features_df[features_df['Feature'].str.startswith('Location_')]

# Plot for 'location_' features
plt.figure(figsize=(12, 8))
bars = plt.barh(location_features_df['Feature'], location_features_df['Importance'], color=colors[:len(location_features_df)])
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Location Feature', fontsize=14)
plt.title('Feature Importance for Location Features', fontsize=16)
plt.gca().invert_yaxis()  # To display the most important features on top
plt.show()

# Filter features for 'Year', 'Month', and 'Day'
date_features_df = features_df[features_df['Feature'].str.contains('Year|Month|Day')]

# Plot for 'Year', 'Month', and 'Day' features
plt.figure(figsize=(12, 8))
bars = plt.barh(date_features_df['Feature'], date_features_df['Importance'], color=colors[:len(date_features_df)])
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Date Feature', fontsize=14)
plt.title('Feature Importance for Year, Month, and Day Features', fontsize=16)
plt.gca().invert_yaxis()  # To display the most important features on top
plt.show()