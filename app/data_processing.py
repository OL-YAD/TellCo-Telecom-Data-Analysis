import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# calculate percentage of missing values
def calculate_missing_percentage(dataframe):
    # Determine the total number of elements in the DataFrame
    total_elements = np.prod(dataframe.shape)

    # Count the number of missing values in each column
    missing_values = dataframe.isna().sum()

    # Sum the total number of missing values
    total_missing = missing_values.sum()

    # Compute the percentage of missing values
    percentage_missing = (total_missing / total_elements) * 100

    # Print the result, rounded to two decimal places
    print(f"The dataset has {round(percentage_missing, 2)}% missing values.")


# display missing values 
def display_missing_values(dataframe):
    # Calculate total missing values per column
    missing_values = dataframe.isnull().sum()

    # Calculate percentage of missing values
    missing_percentage = 100 * missing_values / len(dataframe)

    # Get the data types of the columns
    column_data_types = dataframe.dtypes

    # Combine the results into a DataFrame
    missing_info = pd.concat([missing_values, missing_percentage, column_data_types], axis=1)

    # Rename columns for clarity
    missing_info.columns = ['Missing Values', '% of Total Values', 'Data Type']

    # Filter and sort the table to show only columns with missing values, sorted by percentage
    missing_info_filtered = missing_info[missing_info['Missing Values'] > 0].sort_values(
        '% of Total Values', ascending=False).round(2)

    # Display summary information
    print(f"The dataframe contains {dataframe.shape[1]} columns.\n"
          f"{missing_info_filtered.shape[0]} columns have missing values.")

    # Return the missing values table if there are any missing values
    if missing_info_filtered.shape[0] > 0:
        return missing_info_filtered
    

# A function for handling outliers in specified columns using either iqr or z-score method
def handle_outliers(df, columns, method='iqr', threshold=1.5):
    df_clean = df.copy()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df_clean[col] = df_clean[col].clip(mean - threshold * std, mean + threshold * std)
            else:
                raise ValueError("Method must be either 'iqr' or 'zscore'")
        else:
            print(f"Column {col} is not numeric. Skipping outlier handling for this column.")
    
    return df_clean


def get_top_handsets(df, n=10):
    """Get the top n handsets used by customers."""
    return df['Handset Type'].value_counts().nlargest(n)

def get_top_manufacturers(df, n=3):
    """Get the top n handset manufacturers."""
    return df['Handset Manufacturer'].value_counts().nlargest(n)

def get_top_handsets_per_manufacturer(df, manufacturers, n=5):
    """Get the top n handsets for each of the specified manufacturers."""
    result = {}
    for manufacturer in manufacturers:
        top_handsets = df[df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().nlargest(n)
        result[manufacturer] = top_handsets
    return result

def aggregate_user_behavior(df):
    """Aggregate user behavior on different applications."""
    # Check if the required columns exist
    required_columns = ['MSISDN/Number', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Aggregate basic metrics
    user_behavior = df.groupby('MSISDN/Number').agg({
        'MSISDN/Number': 'count',
        'Dur. (ms)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).rename(columns={
        'MSISDN/Number': 'number_of_xDR_sessions',
        'Dur. (ms)': 'session_duration',
        'Total DL (Bytes)': 'total_DL',
        'Total UL (Bytes)': 'total_UL'
    })
    
    user_behavior['total_data_volume'] = user_behavior['total_DL'] + user_behavior['total_UL']
    
    # Aggregate data for each application
    apps = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
    for app in apps:
        dl_col = f'{app} DL (Bytes)'
        ul_col = f'{app} UL (Bytes)'
        if dl_col in df.columns and ul_col in df.columns:
            app_data = df.groupby('MSISDN/Number')[[dl_col, ul_col]].sum()
            user_behavior[f'{app}_data'] = app_data[dl_col] + app_data[ul_col]
        else:
            print(f"Warning: Columns for {app} not found. Skipping this application.")
    
    return user_behavior

def describe_variables(df):
    """Describe all relevant variables and associated data types."""
    return pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': df.nunique()
    })

def segment_users(df, n_segments=5):
    """Segment users into decile classes based on total duration and compute total data."""
    df['decile_class'] = pd.qcut(df['session_duration'], q=n_segments, labels=False)
    df['total_data'] = df['total_DL'] + df['total_UL']
    return df.groupby('decile_class')['total_data'].sum().reset_index()

def analyze_basic_metrics(df):
    """Analyze basic metrics of the dataset."""
    return df.describe()

def compute_dispersion(df):
    """Compute dispersion parameters for quantitative variables."""
    return df.agg(['var', 'std', 'sem'])
def compute_dispersion(df):
    """Compute dispersion parameters for quantitative variables."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.agg(['var', 'std', 'sem'])

def interpret_dispersion(dispersion_df):
    """Provide interpretation of dispersion parameters."""
    interpretations = []
    for variable in dispersion_df.index:
        var = dispersion_df.loc[variable, 'var']
        std = dispersion_df.loc[variable, 'std']
        sem = dispersion_df.loc[variable, 'sem']
        interpretations.append(f"{variable}: Variance={var:.2f}, Std Dev={std:.2f}, SEM={sem:.2f}")
    return "\n".join(interpretations)


def plot_univariate(df, column):
    """Plot appropriate univariate visualization based on data type."""
    if df[column].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
    else:
        plt.figure(figsize=(12, 6))
        df[column].value_counts().plot(kind='bar')
        plt.title(f'Frequency of {column}')
    plt.show()


def bivariate_analysis(df, app_columns, total_data_column):
    """Explore relationship between each application and total data."""
    correlations = {}
    for app in app_columns:
        corr = df[app].corr(df[total_data_column])
        correlations[app] = corr
    return pd.Series(correlations)

#def correlation_analysis(df, variables):
 #   """Compute correlation matrix for specified variables."""
  #  return df[variables].corr()


def perform_pca(df, n_components=2):
    """Perform PCA on the numeric columns of the DataFrame."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    return pca_result, pca.explained_variance_ratio_

def interpret_pca(explained_variance_ratio):
    """Interpret PCA results."""
    total_variance = sum(explained_variance_ratio)
    interpretations = [
        f"The first component explains {explained_variance_ratio[0]*100:.2f}% of the total variance.",
        f"The second component explains an additional {explained_variance_ratio[1]*100:.2f}% of the variance.",
        f"Together, they explain {total_variance*100:.2f}% of the total variance in the data.",
        "This suggests that these two components capture a significant amount of the variation in the original features."
    ]
    return interpretations
    


    

