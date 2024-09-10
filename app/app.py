import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, sys 
sys.path.append(os.path.abspath(os.path.join('..')))
# Import functions from your script file
from scripts.data_processing import (
    get_top_handsets, get_top_manufacturers, get_top_handsets_per_manufacturer,
    aggregate_user_behavior, segment_users, perform_pca
)
#correlation_analysis
# Load your data
@st.cache_data
def load_data():
    # Replace this with your actual data loading logic
    df = pd.read_csv("../data/cleaned_telecom_data.csv")
    return df

df = load_data()

st.title("Telecom Data Analysis Dashboard")

# Sidebar for task selection
task = st.sidebar.selectbox("Select Task", ["Task 1: User Overview", "Task 2: User Engagement"])

if task == "Task 1: User Overview":
        st.header("Task 1: User Overview Analysis")

        st.header("Task 1: User Overview Analysis")

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

        # 6. Univariate distribution
        st.subheader("6. Univariate Distribution")
        numeric_columns = user_behavior.select_dtypes(include=[np.number]).columns
        selected_column = st.selectbox("Select a column for univariate analysis", numeric_columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(user_behavior[selected_column], kde=True, ax=ax)
        plt.title(f'Distribution of {selected_column}')
        st.pyplot(fig)

        # 7. Bivariate distribution
        st.subheader("7. Bivariate Distribution")
        x_column = st.selectbox("Select X-axis column", numeric_columns, key="x_column")
        y_column = st.selectbox("Select Y-axis column", numeric_columns, key="y_column")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_column, y=y_column, data=user_behavior, ax=ax)
        plt.title(f'{x_column} vs {y_column}')
        st.pyplot(fig)

        # 8. Correlation heatmap
        st.subheader("8. Correlation Heatmap")
        app_columns = ['Social Media_data', 'Google_data', 'Email_data', 'Youtube_data', 'Netflix_data', 'Gaming_data', 'Other_data']
        correlation_matrix = correlation_analysis(user_behavior, app_columns)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Matrix of Application Data Usage')
        st.pyplot(fig)

        # 9. Principal Component Analysis
        st.subheader("9. Principal Component Analysis")
        pca_result, explained_variance = perform_pca(user_behavior[app_columns])
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        st.pyplot(fig)

        st.write("This Streamlit app shows the visualizations for Task 1. You can add more tasks and visualizations as needed.")

elif task == "Task 2: User Engagement":
    st.header("Task 2: User Engagement Analysis")

    # Aggregate metrics per customer
    @st.cache_data
    def aggregate_metrics_per_customer(df):
        aggregated_data = df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',
            'Dur. (ms)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).rename(columns={
            'Bearer Id': 'Sessions',
            'Dur. (ms)': 'Duration',
            'Total DL (Bytes)': 'Download',
            'Total UL (Bytes)': 'Upload'
        })
        aggregated_data['Total Traffic'] = aggregated_data['Download'] + aggregated_data['Upload']
        
        top_10_customers = {
            'Sessions': aggregated_data.nlargest(10, 'Sessions'),
            'Duration': aggregated_data.nlargest(10, 'Duration'),
            'Total Traffic': aggregated_data.nlargest(10, 'Total Traffic')
        }
        
        return aggregated_data, top_10_customers

    aggregated_data, top_10_customers = aggregate_metrics_per_customer(df)

    st.subheader("Top 10 Customers by Engagement Metrics")
    metric = st.selectbox("Select Metric", ["Sessions", "Duration", "Total Traffic"])
    st.dataframe(top_10_customers[metric])

    # Normalize and cluster
    @st.cache_data
    def normalize_and_cluster(aggregated_data):
        normalized_data = scale_and_normalize(aggregated_data, ['Sessions', 'Duration', 'Total Traffic'])
        kmeans = KMeans(n_clusters=3, random_state=42)
        normalized_data['Cluster'] = kmeans.fit_predict(normalized_data)
        return normalized_data, kmeans

    normalized_data, kmeans = normalize_and_cluster(aggregated_data)

    st.subheader("K-means Clustering (k=3)")
    st.write("Normalized data with cluster assignments:")
    st.dataframe(normalized_data)

    # Compute cluster statistics
    @st.cache_data
    def compute_cluster_stats(aggregated_data, normalized_data):
        aggregated_data['Cluster'] = normalized_data['Cluster']
        return aggregated_data.groupby('Cluster').agg(['min', 'max', 'mean', 'sum'])

    cluster_stats = compute_cluster_stats(aggregated_data, normalized_data)
    st.subheader("Cluster Statistics")
    st.dataframe(cluster_stats)

    # Visualize cluster statistics
    st.subheader("Average Metrics per Cluster")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = ['Sessions', 'Duration', 'Total Traffic']

    for i, metric in enumerate(metrics):
        cluster_stats[metric]['mean'].plot(kind='bar', ax=axes[i], title=f'Average {metric} per Cluster')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(f'Average {metric}')

    plt.tight_layout()
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

    # Elbow method for optimal k
    @st.cache_data
    def elbow_method(aggregated_data):
        data = scale_and_normalize(aggregated_data, ['Sessions', 'Duration', 'Total Traffic'])
        inertias = []
        k_range = range(1, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, inertias, 'bx-')
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        return fig

    st.subheader("Elbow Method for Optimal k")
    elbow_fig = elbow_method(aggregated_data)
    st.pyplot(elbow_fig)

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

    # Interpretation and recommendations

st.sidebar.info("Select a task from the dropdown menu to view different analyses.")