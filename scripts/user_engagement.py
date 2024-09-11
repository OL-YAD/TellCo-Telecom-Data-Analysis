import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def aggregate_metrics_per_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per customer (MSISDN) and return top 10 customers per metric.
    """
    aggregated = df.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # sessions frequency
        'Dur. (ms)': 'sum',    # total duration
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).reset_index()
    
    aggregated['Total Traffic'] = aggregated['Total DL (Bytes)'] + aggregated['Total UL (Bytes)']
    aggregated.columns = ['MSISDN', 'Sessions', 'Duration', 'DL Traffic', 'UL Traffic', 'Total Traffic']
    
    top_10 = {
        'Sessions': aggregated.nlargest(10, 'Sessions'),
        'Duration': aggregated.nlargest(10, 'Duration'),
        'Total Traffic': aggregated.nlargest(10, 'Total Traffic')
    }
    
    return aggregated, top_10

def normalize_and_cluster(df: pd.DataFrame, k: int = 3) -> Tuple[pd.DataFrame, KMeans]:
    """
    Normalize engagement metrics and run k-means clustering.
    """
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[['Sessions', 'Duration', 'Total Traffic']]),
                                 columns=['Sessions', 'Duration', 'Total Traffic'])
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    normalized_df['Cluster'] = kmeans.fit_predict(normalized_df)
    
    return normalized_df, kmeans

def compute_cluster_stats(df: pd.DataFrame, normalized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute minimum, maximum, average & total non-normalized metrics for each cluster.
    """
    df['Cluster'] = normalized_df['Cluster']
    return df.groupby('Cluster').agg({
        'Sessions': ['min', 'max', 'mean', 'sum'],
        'Duration': ['min', 'max', 'mean', 'sum'],
        'Total Traffic': ['min', 'max', 'mean', 'sum']
    })

def aggregate_traffic_per_app(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Aggregate user total traffic per application and derive top 10 most engaged users per app.
    """
    apps = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    top_10_per_app = {}
    
    for app in apps:
        df[f'{app} Total'] = df[f'{app} DL (Bytes)'] + df[f'{app} UL (Bytes)']
        top_10 = df.groupby('MSISDN/Number')[f'{app} Total'].sum().nlargest(10).reset_index()
        top_10.columns = ['MSISDN', f'{app} Total Traffic']
        top_10_per_app[app] = top_10
    
    return top_10_per_app

def plot_top_apps(df: pd.DataFrame) -> plt.Figure:
    """
    Plot the top 3 most used applications.
    """
    apps = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    total_traffic = [df[f'{app} DL (Bytes)'].sum() + df[f'{app} UL (Bytes)'].sum() for app in apps]
    top_3_apps = sorted(zip(apps, total_traffic), key=lambda x: x[1], reverse=True)[:3]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([app[0] for app in top_3_apps], [app[1] for app in top_3_apps])
    ax.set_title('Top 3 Most Used Applications')
    ax.set_xlabel('Application')
    ax.set_ylabel('Total Traffic (Bytes)')
    
    return fig

def elbow_method(df: pd.DataFrame, max_k: int = 10) -> plt.Figure:
    """
    Perform elbow method to find optimal k for k-means clustering.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[['Sessions', 'Duration', 'Total Traffic']])
    
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_data)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, max_k + 1), inertias, marker='o')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    

    return fig


