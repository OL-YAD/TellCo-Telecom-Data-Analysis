import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Construct the database URL from environment variables.
def get_database_url():

    db_user = os.getenv('DB_USER')
    db_pass = os.getenv('DB_PASS')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')


    #print(f"DB_USER: {db_user}")
    #print(f"DB_PASS: {'*' * len(db_pass) if db_pass else 'None'}")
    #print(f"DB_HOST: {db_host}")
    #print(f"DB_PORT: {db_port}")
    #print(f"DB_NAME: {db_name}")
    
    return f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Load xdr data from the database.
def load_data(limit=None):
    engine = connect_to_database()
    if not engine:
        return None
    
    query = "SELECT * FROM xdr_data"
    if limit:
        query += f" LIMIT {limit}"
    
    try:
        df = pd.read_sql(query, engine)
        print(f"Successfully loaded {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading xDR records: {e}")
        return None

# Create a connection to the PostgreSQL database using credentials from environment variables.
def connect_to_database():

    database_url = get_database_url()
    try:
        engine = create_engine(database_url) # create sqlalchemy engine
        print("Successfully connected to the database.")
        return engine
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None


# Usage example
if __name__ == "__main__":
    df = load_data(limit=1000)  
    if df is not None:
        print(df.head())