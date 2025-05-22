import pandas as pd
from sqlalchemy import create_engine

# Replace these with your actual database credentials
db_user = 'mlops_user'
db_password = '12345678'
db_host = '172.21.80.1'
db_port = '5432'
db_name = 'mlops'
table_name = 'twitter_comments'

# Path to your CSV file
csv_file_path = '/mnt/d/python/MLOps/clone/MLOPS/data/comments_labeled.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Create a connection to the PostgreSQL database
engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Push the DataFrame to PostgreSQL (replace if table exists)
df.to_sql(table_name, engine, if_exists='replace', index=False)

print("CSV file has been pushed to PostgreSQL!")