import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_customer_data(df):
    """
    Aggregate data per customer, treating missing values and outliers.
    """
    # Group by MSISDN/Number (customer ID)
    grouped = df.groupby('MSISDN/Number')
    
    aggregated = grouped.agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Handset Type': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean'
    })
    
    # Rename columns for clarity
    aggregated.columns = [
        'Avg TCP Retrans DL', 'Avg TCP Retrans UL', 'Avg RTT DL', 'Avg RTT UL',
        'Handset Type', 'Avg Throughput DL', 'Avg Throughput UL'
    ]
    
    # Handle missing values and outliers
    for col in aggregated.columns:
        if col != 'Handset Type':
            aggregated[col] = aggregated[col].fillna(aggregated[col].mean())
            
    # Calculate average TCP retransmission
    aggregated['Avg TCP Retrans'] = (aggregated['Avg TCP Retrans DL'] + aggregated['Avg TCP Retrans UL']) / 2
    
    # Calculate average RTT
    aggregated['Avg RTT'] = (aggregated['Avg RTT DL'] + aggregated['Avg RTT UL']) / 2
    
    # Calculate average throughput
    aggregated['Avg Throughput'] = (aggregated['Avg Throughput DL'] + aggregated['Avg Throughput UL']) / 2
    
    return aggregated

def get_top_bottom_frequent(df, column, n=10):
    """
    Get top, bottom, and most frequent n values for a given column.
    """
    top_n = df[column].nlargest(n).tolist()
    bottom_n = df[column].nsmallest(n).tolist()
    most_frequent = df[column].value_counts().nlargest(n).index.tolist()
    
    return top_n, bottom_n, most_frequent

def plot_distribution(df, x, y, title):
    """
    Plot distribution of y per x.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def perform_kmeans_clustering(df, k=3):
    """
    Perform k-means clustering on the experience metrics.
    """
    # Select features for clustering
    features = ['Avg TCP Retrans', 'Avg RTT', 'Avg Throughput']
    X = df[features]
    
    # Normalize the features
    X_normalized = (X - X.mean()) / X.std()
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_normalized)
    
    return df, kmeans.cluster_centers_

def describe_clusters(df, cluster_centers, features):
    """
    Provide a description of each cluster.
    """
    descriptions = []
    for i, center in enumerate(cluster_centers):
        cluster_df = df[df['Cluster'] == i]
        desc = f"Cluster {i} ({len(cluster_df)} customers):\n"
        for j, feature in enumerate(features):
            desc += f"  - {feature}: {center[j]:.2f} (Avg: {cluster_df[feature].mean():.2f})\n"
        descriptions.append(desc)
    return descriptions