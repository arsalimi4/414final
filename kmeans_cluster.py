import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'amazon.csv'
df = pd.read_csv(file_path)

# Preprocess the data: convert necessary columns into numerical format
df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
df['rating'] = df['rating'].replace('|', None).astype(float)
df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)

# Select relevant columns (excluding 'product_name', 'product_id', 'actual_price', and 'discounted_price')
features_all_metrics = df[['discount_percentage', 'rating', 'rating_count', 'category']].dropna()

# Convert categorical variable 'category' to numerical using one-hot encoding
features_all_metrics = pd.get_dummies(features_all_metrics, columns=['category'])

# Standardize the data
scaler = StandardScaler()
scaled_features_all_metrics = scaler.fit_transform(features_all_metrics)

# Determine the optimal number of clusters using the elbow method
def plot_elbow_method(data):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

# Plot elbow method
plot_elbow_method(scaled_features_all_metrics)

# Perform K-means clustering with the optimal number of clusters (e.g., 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_cleaned = df.dropna(subset=['discount_percentage', 'rating', 'rating_count', 'category'])
df_cleaned['cluster'] = kmeans.fit_predict(scaled_features_all_metrics)

# Analyze the clusters
cluster_summary = df_cleaned.groupby('cluster').agg({
    'discount_percentage': ['mean', 'std'],
    'rating': ['mean', 'std'],
    'rating_count': ['mean', 'std'],
    'category': lambda x: x.mode()[0]  # Most frequent category
})

# Print the cluster summary
print(cluster_summary)