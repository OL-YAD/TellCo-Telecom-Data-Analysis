import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, sys 

sys.path.append(os.path.abspath(os.path.join('..')))


# Import functions from script file
from scripts.data_processing import *
from scripts.user_engagement import *
from scripts.experience_analytics import *


# Load your data
@st.cache_data
def load_data():
    
    
    df = pd.read_csv("../data/cleaned_telecom_data.csv")
    
    return df

df = load_data()

st.title("Telecom Data Analysis Dashboard")
st.write(" Note: This Streamlit app shows the some visualizations from  the analysis. for more refer to the notebook file.")

# Sidebar for task selection
task = st.sidebar.selectbox("Select Task", ["User Overview Analysis", "User Engagement Analysis","User Experience Analytics"])

if task == "User Overview Analysis":
        st.header(" User Overview Analysis")


        # 1. Data field description
        st.subheader("1. Data Field Description")
        st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Field'}))

        # 2. Top 10 handsets
        st.subheader("2. Top 10 Handsets Used by Customers")
        top_10_handsets = get_top_handsets(df)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index, ax=ax)
        plt.title('Top 10 Handsets Used by Customers')
        plt.xlabel('Number of Handsets')
        plt.ylabel('Handset Type')
        st.pyplot(fig)

        # 3. Top 3 handset manufacturers
        st.subheader("3. Top 3 Handset Manufacturers")
        top_3_manufacturers = get_top_manufacturers(df)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_3_manufacturers.values, y=top_3_manufacturers.index, ax=ax)
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Number of Handsets')
        plt.ylabel('Manufacturer')
        st.pyplot(fig)

        # 4. Top 5 handsets per top 3 handset manufacturer
        st.subheader("4. Top 5 Handsets per Top 3 Handset Manufacturer")
        top_5_handsets_per_manufacturer = get_top_handsets_per_manufacturer(df, top_3_manufacturers.index)
        for manufacturer, handsets in top_5_handsets_per_manufacturer.items():
            st.write(f"**{manufacturer}**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=handsets.values, y=handsets.index, ax=ax)
            plt.title(f'Top 5 Handsets for {manufacturer}')
            plt.xlabel('Number of Handsets')
            plt.ylabel('Handset Type')
            st.pyplot(fig)

        # 5. Total data usage per decile class
        st.subheader("5. Total Data Usage per Decile Class")
        user_behavior = aggregate_user_behavior(df)
        user_behavior_segmented = segment_users(user_behavior)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='decile_class', y='total_data', data=user_behavior_segmented, ax=ax)
        plt.title('Total Data Usage per Decile Class')
        plt.xlabel('Decile Class')
        plt.ylabel('Total Data Usage')
        st.pyplot(fig)

        # 6. Correlation heatmap
        st.subheader("6. Correlation Matrix of Application Data Usage ")
        df["Youtube_Total_Data"] = df["Youtube DL (Bytes)"] + df["Youtube UL (Bytes)"]
        df["Google_Total_Data"] = df["Google DL (Bytes)"] + df["Google UL (Bytes)"]
        df["Email_Total_Data"] = df["Email DL (Bytes)"] + df["Email UL (Bytes)"]
        df["Social_Media_Total_Data"] = df["Social Media DL (Bytes)"] + df["Social Media UL (Bytes)"]
        df["Netflix_Total_Data"] = df["Netflix DL (Bytes)"] + df["Netflix UL (Bytes)"]
        df["Gaming_Total_Data"] = df["Gaming DL (Bytes)"] + df["Gaming UL (Bytes)"]
        df["Other_Total_Data"] = df["Other DL (Bytes)"] + df["Other UL (Bytes)"]
        df["Total_UL_and_DL"] = df["Total UL (Bytes)"] + df["Total DL (Bytes)"]


        df_corr = df[['Youtube_Total_Data', 'Google_Total_Data','Email_Total_Data', 'Social_Media_Total_Data',
              'Netflix_Total_Data','Gaming_Total_Data','Other_Total_Data']].corr()
        
        fig,ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Application Data Usage')
        st.pyplot(fig)


        st.write("This Streamlit app shows the some visualizations from Task 1.")
        st.sidebar.info("Select a task from the dropdown menu to view different analyses.")

elif task == "User Engagement Analysis":
    st.header("User Engagement Analysis")

    
    # Compute cluster statistics
   
    #aggregrate mmetrics per customer 
    aggregated_data, top_10_customers = aggregate_metrics_per_customer(df)
    st.subheader("Top 10 Customer by number of session")
    print("Top 10 customers by number of sessions:")
    print(top_10_customers['Sessions'])

    normalized_data, kmeans = normalize_and_cluster(aggregated_data)
    # Compute cluster statistics
    st.subheader("Cluster statistics")
    cluster_stats = compute_cluster_stats(aggregated_data, normalized_data)
    print("\nCluster statistics:")
    cluster_stats

    # Visualize cluster statistics
    st.subheader("Visualization of Cluster statistics")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = ['Sessions', 'Duration', 'Total Traffic']

    for i, metric in enumerate(metrics):
        cluster_stats[metric]['mean'].plot(kind='bar', ax=axes[i], title=f'Average {metric} per Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(f'Average {metric}')

    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
    # Aggregate user total traffic per application
    @st.cache_data
    def aggregate_traffic_per_app(df):
        apps = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
        top_10_per_app = {}
        
        for app in apps:
            app_traffic = df.groupby('MSISDN/Number')[f'{app} DL (Bytes)'].sum() + df.groupby('MSISDN/Number')[f'{app} UL (Bytes)'].sum()
            top_10_per_app[app] = app_traffic.nlargest(10)
        
        return top_10_per_app

    top_10_per_app = aggregate_traffic_per_app(df)

    st.subheader("Top 10 Users per Application")
    app = st.selectbox("Select Application", ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other'])
    st.dataframe(top_10_per_app[app])

    # Visualize application usage distribution
    st.subheader("Application Usage Distribution")
    apps = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    app_usage = [df[f'{app} DL (Bytes)'].sum() + df[f'{app} UL (Bytes)'].sum() for app in apps]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(app_usage, labels=apps, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Plot top 3 most used applications
    @st.cache_data
    def plot_top_apps(df):
        apps = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
        app_usage = [df[f'{app} DL (Bytes)'].sum() + df[f'{app} UL (Bytes)'].sum() for app in apps]
        top_3_apps = sorted(zip(apps, app_usage), key=lambda x: x[1], reverse=True)[:3]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([app[0] for app in top_3_apps], [app[1] for app in top_3_apps])
        ax.set_title('Top 3 Most Used Applications')
        ax.set_xlabel('Application')
        ax.set_ylabel('Total Traffic (Bytes)')
        return fig

    st.subheader("Top 3 Most Used Applications")
    top_apps_fig = plot_top_apps(df)
    st.pyplot(top_apps_fig)

    # Engagement patterns over time
    st.subheader("Engagement Patterns Over Time")
    df['Start'] = pd.to_datetime(df['Start'])
    df['Hour'] = df['Start'].dt.hour
    df['Day'] = df['Start'].dt.day_name()

    hourly_engagement = df.groupby('Hour')['Dur. (ms)'].mean()
    daily_engagement = df.groupby('Day')['Dur. (ms)'].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    hourly_engagement.plot(kind='bar', ax=ax1)
    ax1.set_title('Average Session Duration by Hour')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Average Duration (ms)')

    daily_engagement.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Session Duration by Day')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Average Duration (ms)')

    plt.tight_layout()
    st.pyplot(fig)

    st.sidebar.info("Select a task from the dropdown menu to view different sample analyses.")


elif task == "User Experience Analytics":
    st.header("Experience Analytics")

    # Aggregate customer data
    aggregated_data = aggregate_customer_data(df)

    st.subheader("Aggregated Customer Data")
    st.dataframe(aggregated_data.head())

    # Top, bottom, and most frequent values
    metrics = ['Avg TCP Retrans', 'Avg RTT', 'Avg Throughput']
    selected_metric = st.selectbox("Select Metric", metrics)
    top, bottom, frequent = get_top_bottom_frequent(aggregated_data, selected_metric)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Top 10")
        st.write(top)
    with col2:
        st.write("Bottom 10")
        st.write(bottom)
    with col3:
        st.write("Most Frequent 10")
        st.write(frequent)

    # Distribution plots
    st.subheader("Distribution Plots")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Handset Type', y=selected_metric, data=aggregated_data, ax=ax)
    plt.title(f'{selected_metric} Distribution per Handset Type')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # K-means clustering
    st.subheader("K-means Clustering")
    clustered_data, cluster_centers = perform_kmeans_clustering(aggregated_data)
    
    features = ['Avg TCP Retrans', 'Avg RTT', 'Avg Throughput']
    cluster_descriptions = describe_clusters(clustered_data, cluster_centers, features)
    
    for desc in cluster_descriptions:
        st.text(desc)

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=clustered_data, x='Avg Throughput', y='Avg RTT', hue='Cluster', palette='deep', ax=ax)
    plt.title('Customer Clusters based on Experience Metrics')
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap of Experience Metrics")
    correlation_matrix = clustered_data[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    st.pyplot(fig)

    # Distribution of experience metrics
    st.subheader("Distribution of Experience Metrics")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, feature in enumerate(features):
        sns.histplot(clustered_data[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    st.pyplot(fig)

    # Cluster centroids comparison
    st.subheader("Comparison of Cluster Centroids")
    centroid_df = pd.DataFrame(cluster_centers, columns=features)
    centroid_df = centroid_df.melt(var_name='Metric', value_name='Value')
    centroid_df['Cluster'] = np.repeat(range(3), 3)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Metric', y='Value', hue='Cluster', data=centroid_df, ax=ax)
    st.pyplot(fig)

    # Experience metrics by handset manufacturer
    st.subheader("Experience Metrics by Handset Manufacturer")
    aggregated_data['Handset Manufacturer'] = aggregated_data['Handset Type'].apply(lambda x: x.split()[0])
    manufacturer_metrics = aggregated_data.groupby('Handset Manufacturer')[features].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(manufacturer_metrics, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
    plt.title('Average Experience Metrics by Handset Manufacturer')
    st.pyplot(fig)

    st.sidebar.info("Select a task from the dropdown menu to view different analyses.")
