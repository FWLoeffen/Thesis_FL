import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataw2v = pd.read_csv('df_w2v_skimmed.csv')
datatf = pd.read_csv('df_tfidf_skimmed.csv')

dataw2v['Word2Vec_Cluster_Labels'].unique()

# Group the data by 'Word2Vec_Clusters' and collect the unique labels for each cluster
cluster_label_mapping = dataw2v.groupby('Word2Vec_Clusters')['Word2Vec_Cluster_Labels'].unique().to_dict()

# Print the dictionary to see the mapping
print(cluster_label_mapping)

# Count the occurrences of each cluster label
cluster_counts = dataw2v['Word2Vec_Clusters'].value_counts().reindex(cluster_label_mapping.keys())

# Create a bar plot
plt.figure(figsize=(12, 8))
bars = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')

# Adding labels and title
plt.xlabel('Word2Vec Clusters')
plt.ylabel('Frequency')
plt.title('Distribution of Word2Vec Clusters')

# Set the ticks to show cluster numbers with corresponding labels
plt.xticks()  # Rotate the x labels to make them more readable

# Creating a legend manually
legend_labels = [f'Cluster {cluster}: {", ".join(map(str, labels))}' for cluster, labels in cluster_label_mapping.items()]
colors = plt.cm.viridis(np.linspace(0, 1, len(legend_labels)))  # Generating colors for each cluster

# Create custom patches for the legend
from matplotlib.patches import Patch
legend_handles = [Patch(color=color, label=label) for color, label in zip(colors, legend_labels)]
plt.legend(handles=legend_handles, title="Cluster Mappings", loc='upper right')

# Group the data by 'Word2Vec_Clusters' and collect the unique labels for each cluster
cluster_label_mapping = datatf.groupby('TFIDF_Clusters')['TFIDF_Cluster_Labels'].unique().to_dict()

# Print the dictionary to see the mapping
print(cluster_label_mapping)

# Count the occurrences of each cluster label
cluster_counts = datatf['TFIDF_Clusters'].value_counts().reindex(cluster_label_mapping.keys())

# Create a bar plot
plt.figure(figsize=(12, 8))
bars = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')

# Adding labels and title
plt.xlabel('TF-IDF Clusters')
plt.ylabel('Frequency')
plt.title('Distribution of TF-IDF Clusters')

# Set the ticks to show cluster numbers with corresponding labels
plt.xticks()  # Rotate the x labels to make them more readable

# Creating a legend manually
legend_labels = [f'Cluster {cluster}: {", ".join(map(str, labels))}' for cluster, labels in cluster_label_mapping.items()]
colors = plt.cm.viridis(np.linspace(0, 1, len(legend_labels)))  # Generating colors for each cluster

# Create custom patches for the legend
from matplotlib.patches import Patch
legend_handles = [Patch(color=color, label=label) for color, label in zip(colors, legend_labels)]
plt.legend(handles=legend_handles, title="Cluster Mappings", loc='upper right')

