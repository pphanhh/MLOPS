import pandas as pd
from sqlalchemy import create_engine
import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def connect_to_db(db_name="postgres"):
    """Create database connection using environment variables"""
    db_user = os.getenv("DB_USER", "mlops_user")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "172.21.80.1")
    db_port = os.getenv("DB_PORT", "5432")

    conn_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(conn_string)

def get_latest_labeled_file():
    """Get the latest labeled file from the labeled directory"""
    labeled_dir = '/mnt/d/python/MLOps/clone/MLOPS/data/labeled'
    files = [f for f in os.listdir(labeled_dir) if f.endswith('.csv')]
    if not files:
        return None
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(labeled_dir, x)))
    return os.path.join(labeled_dir, latest_file)

def main():
    """Main function to load data from CSV to PostgreSQL"""
    import argparse

    parser = argparse.ArgumentParser(description="Load CSV data into PostgreSQL")
    parser.add_argument(
        "--file", "-f",
        default=get_latest_labeled_file(),
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--database", "-d",
        default="mlops",
        help="Database name"
    )
    parser.add_argument(
        "--table", "-t",
        default="comments_labeled",
        help="Table name"
    )
    parser.add_argument(
        "--mode", "-m",
        default="append",
        choices=["replace", "append", "fail"],
        help="How to handle existing tables"
    )

    args = parser.parse_args()

    if not args.file or not os.path.exists(args.file):
        logger.error(f"❌ File not found: {args.file}")
        sys.exit(1)

    try:
        df = pd.read_csv(args.file)
        df.columns = [col.lower() for col in df.columns]
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'])
        logger.info(f"✅ Đã đọc {len(df)} dòng từ file: {args.file}")
    except Exception as e:
        logger.error(f"❌ Lỗi khi đọc file CSV: {e}")
        sys.exit(1)

    try:
        engine = connect_to_db(args.database)
        if 'id' in df.columns:
            # Fetch existing IDs from the database
            with engine.connect() as conn:
                existing_ids = pd.read_sql(f"SELECT id FROM {args.table}", conn)
            # Remove rows with IDs already in the database
            df = df[~df['id'].isin(existing_ids['id'])]
            logger.info(f"✅ Đã loại bỏ {len(existing_ids)} bản ghi trùng id với DB, còn lại {len(df)} bản ghi để ghi vào DB.")
        df.to_sql(args.table, engine, if_exists=args.mode, index=False)
        logger.info(f"✅ Đã ghi {len(df)} dòng vào {args.database}.{args.table}")
    except Exception as e:
        logger.error(f"❌ Lỗi khi ghi dữ liệu vào PostgreSQL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
