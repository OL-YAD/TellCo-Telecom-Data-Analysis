import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, List
import mysql.connector

def calculate_euclidean_distance(point: np.ndarray, centroid: np.ndarray) -> float:
    """Calculate Euclidean distance between a point and a centroid."""
    return np.sqrt(np.sum((point - centroid)**2))

    
def assign_scores(engagement_data: pd.DataFrame, experience_data: pd.DataFrame, 
                  engagement_kmeans: KMeans, experience_kmeans: KMeans,
                  msisdn_column: pd.Series) -> pd.DataFrame:
    """
    Assign engagement and experience scores to each user.
    """
    # Normalize data
    scaler = StandardScaler()
    engagement_normalized = scaler.fit_transform(engagement_data[['Sessions', 'Duration', 'Total Traffic']])
    experience_normalized = scaler.fit_transform(experience_data[['Avg TCP Retrans', 'Avg RTT', 'Avg Throughput']])
    
    # Find least engaged and worst experience centroids
    least_engaged_centroid = engagement_kmeans.cluster_centers_[np.argmin(np.sum(engagement_kmeans.cluster_centers_, axis=1))]
    worst_experience_centroid = experience_kmeans.cluster_centers_[np.argmin(np.sum(experience_kmeans.cluster_centers_, axis=1))]
    
    # Calculate scores
    engagement_scores = np.array([calculate_euclidean_distance(point, least_engaged_centroid) 
                                  for point in engagement_normalized])
    experience_scores = np.array([calculate_euclidean_distance(point, worst_experience_centroid) 
                                  for point in experience_normalized])
    
    # Create DataFrame with scores
    scores_df = pd.DataFrame({
        'MSISDN': msisdn_column.unique(),
        'Engagement Score': engagement_scores,
        'Experience Score': experience_scores
    })
    
    return scores_df

def calculate_satisfaction_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate satisfaction scores and return top 10 satisfied customers.
    
    :param scores_df: DataFrame with engagement and experience scores
    :return: DataFrame with satisfaction scores and top 10 satisfied customers
    """
    scores_df['Satisfaction Score'] = (scores_df['Engagement Score'] + scores_df['Experience Score']) / 2
    top_10_satisfied = scores_df.nlargest(10, 'Satisfaction Score')
    return scores_df, top_10_satisfied

def build_regression_model(scores_df: pd.DataFrame) -> Tuple[LinearRegression, float, float]:
    """
    Build a regression model to predict satisfaction scores.
    
    :param scores_df: DataFrame with engagement, experience, and satisfaction scores
    :return: Tuple of (model, MSE, R2 score)
    """
    X = scores_df[['Engagement Score', 'Experience Score']]
    y = scores_df['Satisfaction Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

def cluster_satisfaction(scores_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[float], List[float]]:
    """
    Perform k-means clustering (k=2) on engagement and experience scores.
    
    :param scores_df: DataFrame with engagement and experience scores
    :return: Tuple of (DataFrame with cluster assignments, avg satisfaction per cluster, avg experience per cluster)
    """
    X = scores_df[['Engagement Score', 'Experience Score']]
    kmeans = KMeans(n_clusters=2, random_state=42)
    scores_df['Cluster'] = kmeans.fit_predict(X)
    
    avg_satisfaction = scores_df.groupby('Cluster')['Satisfaction Score'].mean().tolist()
    avg_experience = scores_df.groupby('Cluster')['Experience Score'].mean().tolist()
    
    return scores_df, avg_satisfaction, avg_experience

def plot_clusters(scores_df: pd.DataFrame) -> plt.Figure:
    """
    Plot the clusters of users based on their engagement and experience scores.
    
    :param scores_df: DataFrame with engagement, experience scores, and cluster assignments
    :return: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(scores_df['Engagement Score'], scores_df['Experience Score'], 
                         c=scores_df['Cluster'], cmap='viridis')
    ax.set_xlabel('Engagement Score')
    ax.set_ylabel('Experience Score')
    ax.set_title('User Clusters based on Engagement and Experience')
    plt.colorbar(scatter)
    return fig


# function for exporting to table to MySQL

def export_to_mysql(scores_df: pd.DataFrame, db_name: str, table_name: str):
    """
    Export the final table to a local MySQL database.
    
    :param scores_df: DataFrame with user IDs, engagement, experience, and satisfaction scores
    :param db_name: Name of the MySQL database
    :param table_name: Name of the table to be created
    """
    # Connect to the MySQL database
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="oloteme"
    )
    
    # Create the database if it doesn't exist
    cursor = db.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    cursor.execute(f"USE {db_name}")
    
    # Create the table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            MSISDN VARCHAR(255),
            Engagement_Score FLOAT,
            Experience_Score FLOAT,
            Satisfaction_Score FLOAT
        )
    """)
    
    # Insert the data into the table
    for _, row in scores_df.iterrows():
        sql = f"INSERT INTO {table_name} (MSISDN, Engagement_Score, Experience_Score, Satisfaction_Score) VALUES (%s, %s, %s, %s)"
        values = (row['MSISDN'], row['Engagement Score'], row['Experience Score'], row['Satisfaction Score'])
        cursor.execute(sql, values)
    
    db.commit()
    db.close()


def read_from_mysql(db_name: str, table_name: str) -> pd.DataFrame:
    """
    Read data from a MySQL database table and return it as a pandas DataFrame.
    
    :param db_name: Name of the MySQL database
    :param table_name: Name of the table to read from
    :return: DataFrame containing the data from the MySQL table
    """
    try:
        # Connect to the MySQL database
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="oloteme",
            database=db_name
        )
        
        # Create a cursor object
        cursor = db.cursor()
        
        # Execute a SELECT query
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [i[0] for i in cursor.description]
        
        # Create a DataFrame
        df = pd.DataFrame(rows, columns=column_names)
        
        # Close the cursor and database connection
        cursor.close()
        # Close the database connection
        db.close()
        
        return df
    
    except mysql.connector.Error as error:
        print(f"Error reading data from MySQL table: {error}")
        return None

