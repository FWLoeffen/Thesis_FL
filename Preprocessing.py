import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


df_2021 = pd.read_csv('AV_Disengagement_2021.csv', header=0, sep= ',')
df_2022 = pd.read_csv('AV_Disengagement_2022.csv', header=0, sep=',')
df_2023 = pd.read_csv('AV_Disengagement_2023.csv', header=0, sep=',')


#drop the unnamed columns at the end
df_2021 = df_2021.loc[:, ~df_2021.columns.str.contains('^Unnamed')]

# Concatenate the dataframes
df_combined = pd.concat([df_2021, df_2022, df_2023])

# Ensure that the 'date' column is in datetime format if it's not already
df_combined['DATE'] = pd.to_datetime(df_combined['DATE'])

# Sort the dataframe by 'Manufacturer' alphabetically and 'date' chronologically
df_combined = df_combined.sort_values(by=['Manufacturer', 'DATE'])

# Count the number of missing values in each column
missing_values_count = df_combined.isna().sum()
print(missing_values_count)

# drop the operating without driver & Driver present columns, as they add nothing
df_combined = df_combined.drop(df_combined.columns[range(4, 6)], axis=1)

# change colnames
df_combined.columns = ['Manufacturer', 'Permit_number', 'Date', 'VIN_number', 'Disengagement_by', 'Location', 'Cause_of_disengagement']

# change the dtypes to str except for date
df_combined[['Manufacturer', 'Permit_number', 'VIN_number', 'Disengagement_by', 'Location', 'Cause_of_disengagement']] = df_combined[['Manufacturer', 'Permit_number', 'VIN_number', 'Disengagement_by', 'Location', 'Cause_of_disengagement']].astype(str)

#save the file
df_combined.to_csv('df_combined.csv', index=False)

### I lost the file where I cleaned the target feature, which was called 'Clean_CoD_Col.csv'. From there this code continues
# W2v Framework
file_path = 'Clean_CoD_col.csv'
data = pd.read_csv(file_path)

# Download and update NLTK stop words list
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Tokenize the text data: split each entry into a list of words and remove stop words and short words
filtered_sentences = [[word for word in row.split() if word not in stop_words and len(word) > 2] 
                      for row in data['Cause_of_disengagement']]

# Retrain the Word2Vec model with the filtered sentences
filtered_model = Word2Vec(sentences=filtered_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Inspect the first 10 words from the vocabulary of the new model
list(filtered_model.wv.index_to_key)[:10]

# Compute average Word2Vec vector for each document
X = np.array([np.mean([filtered_model.wv[word] for word in words if word in filtered_model.wv.index_to_key], axis=0) 
              for words in filtered_sentences])

# Define a range of possible number of clusters
cluster_range = range(2, 11)

# Calculate Silhouette scores for each number of clusters for Word2Vec
silhouette_scores_word2vec = []
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores_word2vec.append(silhouette_avg)

# Print the silhouette scores to find the optimal number of clusters
print("Silhouette Scores for Word2Vec:", silhouette_scores_word2vec)

# perform Kmeans with optimal clusters
kmeans_word2vec = KMeans(n_clusters=10, random_state=42)
word2vec_clusters = kmeans_word2vec.fit_predict(X)
data['Word2Vec_Clusters'] = word2vec_clusters

def get_representative_docs(X, labels, data, num_docs=5):
    # Calculate the centroids of each cluster
    centroids = np.array([X[labels == k].mean(axis=0) for k in range(10)])
    representative_docs = {}

    # Find the closest documents to each centroid
    for i, centroid in enumerate(centroids):
        # Calculate distances from each document in the cluster to the centroid
        dists = distance.cdist([centroid], X[labels == i], 'cosine')[0]
        # Get the indices of the documents with the smallest distances
        closest_indices = np.argsort(dists)[:num_docs]
        # Retrieve the actual documents
        representative_docs[i] = data.iloc[labels == i].iloc[closest_indices]['Cause_of_disengagement'].values

    return representative_docs

# Get representative documents for each Word2Vec cluster
word2vec_representative_docs = get_representative_docs(X, word2vec_clusters, data)
data['Word2Vec_Clusters'].value_counts()

# From here, Chatgpt analyzes the representative docs, and manually creates class labels

word2vec_cluster_labels = {
    0: "Hardware Malfunctions",
    1: "Lane Placement Errors",
    2: "Predictive System Failures",
    3: "Software System Errors",
    4: "Operational Decision Making",
    5: "Proximity Detection Issues",
    6: "Software Discrepancies",
    7: "Courtesy-Based Disengagements",
    8: "Navigation Errors",
    9: "Control and Responsiveness"
}

# Map these labels to the cluster IDs in the dataframe
data['Word2Vec_Cluster_Labels'] = data['Word2Vec_Clusters'].map(word2vec_cluster_labels)

# load original dataset and add the columns

df_complete = pd.read_csv('Grouped_AVdata.csv', header=0, sep=',')
# Add Word2Vec cluster numbers and labels to the 'df_complete' DataFrame
df_complete['Word2Vec_Clusters'] = data['Word2Vec_Clusters']
df_complete['Word2Vec_Cluster_Labels'] = data['Word2Vec_Cluster_Labels']
df_complete.head()

#save the dataset based on w2v
df_complete.to_csv('df_w2v_ready.csv', index=False)

### Now on to the process for TF-IDF Framework
file_path = 'Clean_CoD_col.csv'
data = pd.read_csv(file_path)

# Download and update NLTK stop words list
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Tokenize the text data: split each entry into a list of words and remove stop words and short words
filtered_sentences = [[word for word in row.split() if word not in stop_words and len(word) > 2] 
                      for row in data['Cause_of_disengagement']]

# Combine the tokens back into strings for TF-IDF vectorization
preprocessed_texts = [" ".join(sentence) for sentence in filtered_sentences]

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limiting to the top 1000 features for simplicity

# Fit and transform the preprocessed texts to a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)

# shape of the TF-IDF matrix and feature names
print(tfidf_matrix.shape)
print(tfidf_vectorizer.get_feature_names_out()[:10])

# Define a range of possible number of clusters
cluster_range = range(2, 11)

# Calculate Silhouette scores for each number of clusters for TF-IDF
silhouette_scores_tfidf = []
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
    silhouette_scores_tfidf.append(silhouette_avg)

# Print the silhouette scores to find the optimal number of clusters
print("Silhouette Scores for TF-IDF:", silhouette_scores_tfidf)

# Perform K-means clustering
kmeans_tfidf = KMeans(n_clusters=9, random_state=42)
tfidf_clusters = kmeans_tfidf.fit_predict(tfidf_matrix)  # tfidf_matrix is your TF-IDF features matrix
data['TFIDF_Clusters'] = tfidf_clusters

def get_top_terms_for_each_cluster(tfidf_vectorizer, kmeans_model, n_terms=10):
    terms = tfidf_vectorizer.get_feature_names_out()
    centroids = kmeans_model.cluster_centers_
    top_terms = {}
    for i in range(9):  # Assuming 9 clusters
        top_term_indices = centroids[i].argsort()[-n_terms:][::-1]
        top_terms[i] = [terms[ind] for ind in top_term_indices]
    return top_terms

# Get the top terms for each TF-IDF cluster
tfidf_top_terms = get_top_terms_for_each_cluster(tfidf_vectorizer, kmeans_tfidf)
data['TFIDF_Clusters'].value_counts()

# Here Chatgpt analyzes the top terms and manually creates labels

tfidf_cluster_labels = {
    0: "Environmental Perception Errors",
    1: "Mapping and Navigation Discrepancies",
    2: "Behavior Prediction Failures",
    3: "System Software Issues",
    4: "Incorrect Maneuvering and Placement",
    5: "Unintended Vehicle Maneuvers",
    6: "Lane Management Under Specific Conditions",
    7: "Hardware Malfunctions",
    8: "Social and Courtesy Expectations"
}

# Map labels to the cluster IDs in the dataframe
data['TFIDF_Cluster_Labels'] = data['TFIDF_Clusters'].map(tfidf_cluster_labels)

#load original dataset and add columns
df_complete = pd.read_csv('Grouped_AVdata.csv', header=0, sep=',')
# Add TF-IDF cluster numbers and labels to the 'df_complete' DataFrame
df_complete['TFIDF_Clusters'] = data['TFIDF_Clusters']
df_complete['TFIDF_Cluster_Labels'] = data['TFIDF_Cluster_Labels']
df_complete.to_csv('df_tfidf_ready.csv', index=False)

### Feature extraction of predictors

# Initialize encoders
one_hot_encoder = OneHotEncoder(sparse=False)

df_tf = pd.read_csv('df_tfidf_ready.csv')
df_w2v = pd.read_csv('df_w2v_ready.csv')

df_tf = df_tf.drop(df_tf[['Permit_number', 'VIN_number', 'Disengagement_by', 'Frequency']], axis=1)
df_w2v = df_w2v.drop(df_w2v[['Permit_number', 'VIN_number', 'Disengagement_by', 'Frequency']], axis=1)

# Applying One-Hot Encoding to 'Manufacturer' & Location
manuf_onehot = ['Manufacturer', 'Location']
manuf_tf_encoded = one_hot_encoder.fit_transform(df_tf[manuf_onehot])
manuf_w2v_encoded = one_hot_encoder.fit_transform(df_w2v[manuf_onehot])

# Use get_feature_names_out() to get the new column names for the encoded features
manuf_tf_encoded_df = pd.DataFrame(manuf_tf_encoded, columns=one_hot_encoder.get_feature_names_out(manuf_onehot))
manuf_w2v_encoded_df = pd.DataFrame(manuf_w2v_encoded, columns=one_hot_encoder.get_feature_names_out(manuf_onehot))

# Combine the original DataFrame with the new one-hot encoded DataFrame
df_tf = pd.concat([df_tf.reset_index(drop=True), manuf_tf_encoded_df.reset_index(drop=True)], axis=1)
df_w2v = pd.concat([df_w2v.reset_index(drop=True), manuf_w2v_encoded_df.reset_index(drop=True)], axis=1)

df_tf.drop(manuf_onehot, axis=1, inplace=True)
df_w2v.drop(manuf_onehot, axis=1, inplace=True)

# Date handling
df_tf['Year'] = pd.to_datetime(df_tf['Date']).dt.year
df_tf['Month'] = pd.to_datetime(df_tf['Date']).dt.month
df_tf['Day'] = pd.to_datetime(df_tf['Date']).dt.day

df_w2v['Year'] = pd.to_datetime(df_w2v['Date']).dt.year
df_w2v['Month'] = pd.to_datetime(df_w2v['Date']).dt.month
df_w2v['Day'] = pd.to_datetime(df_w2v['Date']).dt.day

df_tf.drop('Date', axis=1, inplace=True)
df_w2v.drop('Date', axis=1, inplace=True)

# save files that only contain relevant features
df_tf.to_csv('df_tfidf_skimmed.csv', index=False)
df_w2v.to_csv('df_w2v_skimmed.csv', index=False)

