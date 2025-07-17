"""
Simplified data ingestion module for loading CSV files into PostgreSQL
"""
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
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

def create_database_if_not_exists(db_name):
    """Create the database if it doesn't exist"""
    try:
        # Connect to default postgres database
        engine = connect_to_db("postgres")
        
        # Check if database exists
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
            exists = result.fetchone()
            
            if not exists:
                logger.info(f"Creating database '{db_name}'...")
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                logger.info(f"Database '{db_name}' created.")
            else:
                logger.info(f"Database '{db_name}' already exists.")
                
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def load_data_to_db(csv_file, db_name, table_name, if_exists="replace"):
    """
    Load data from CSV file to PostgreSQL database
    
    Args:
        csv_file: Path to the CSV file
        db_name: Database name
        table_name: Table name
        if_exists: How to handle existing table ('replace', 'append', 'fail')
    
    Returns:
        Number of records loaded
    """
    try:
        # Ensure database exists
        create_database_if_not_exists(db_name)
        
        # Load CSV data
        logger.info(f"Loading data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Convert any date columns to datetime
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Connect to target database and insert data
        engine = connect_to_db(db_name)
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        
        logger.info(f"Loaded {len(df)} records into {db_name}.{table_name}")
        return len(df)
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def get_latest_labeled_file():
        """Get the latest labeled file from the labeled directory"""
        labeled_dir = '/mnt/d/MLOps2/data/labeled'
        files = [f for f in os.listdir(labeled_dir) if f.endswith('.csv')]
        if not files:
            return None
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(labeled_dir, x)))
        return os.path.join(labeled_dir, latest_file)

def test_connection():
    """Test the database connection properly using psycopg2"""
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "host.docker.internal")
    db_port = os.getenv("DB_PORT", "5432")
    
    try:
        # Thử kết nối trực tiếp bằng psycopg2
        conn = psycopg2.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            database="postgres",  # Kết nối đến postgres mặc định
            connect_timeout=3     # Timeout sau 3 giây
        )
        conn.close()
        print(f"Successfully connected to PostgreSQL at {db_host}:{db_port}")
        return True
    except Exception as e:
        print(f"Failed to connect to PostgreSQL at {db_host}:{db_port}")
        print(f"Error message: {str(e)}")
        print("\nGợi ý khắc phục:")
        print("1. Nếu PostgreSQL đang chạy trên Windows và bạn đang sử dụng WSL:")
        print("   - Tạo file .env với nội dung sau:")
        print('     DB_HOST=host.docker.internal  # hoặc IP của Windows')
        print('     DB_USER=postgres')
        print('     DB_PASSWORD=your_password')
        print('     DB_PORT=5432')
        print("   - Đảm bảo PostgreSQL được cấu hình nhận kết nối từ xa:")
        print("     + Sửa file pg_hba.conf để thêm: host all all 0.0.0.0/0 md5")
        print("     + Sửa file postgresql.conf: listen_addresses = '*'")
        print("2. Sử dụng công cụ ingest_sqlite.py thay thế (đơn giản hơn):")
        print("   python ingest_sqlite.py --file your_file.csv")
        return False

# Chạy test_connection() ở đầu hàm main() để kiểm tra
def main():
    """Main function to load data from CSV to PostgreSQL"""
    import argparse
    from datetime import datetime
    
    # Test connection first - if it fails, suggest alternatives
    if not test_connection():
        print("Không thể kết nối đến PostgreSQL. Đang dừng chương trình.")
        sys.exit(1)
    
    # Get default input file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        default_input = get_latest_labeled_file() or f'/mnt/d/MLOps2/data/labeled/labeled_twitter_{timestamp}.csv'
    except Exception as e:
        default_input = f'/mnt/d/MLOps2/data/labeled/labeled_twitter_{timestamp}.csv'
        print(f"Không thể tìm thấy file CSV mặc định: {e}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load CSV data into PostgreSQL")
    parser.add_argument(
        "--file", "-f", 
        default=default_input,
        help="Path to the input CSV file (raw data)."
    )
    parser.add_argument(
        "--database", "-d", 
        default="twitter_analysis", 
        help="Database name"
    )
    parser.add_argument(
        "--table", "-t", 
        default="tweets", 
        help="Table name"
    )
    parser.add_argument(
        "--mode", "-m", 
        default="append", 
        choices=["replace", "append", "fail"],
        help="How to handle existing tables"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
    
    # Use the same connection parameters that worked in test_connection()
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "host.docker.internal")
    db_port = os.getenv("DB_PORT", "5432")
    
    # Load a small piece of data first to test the connection thoroughly
    try:
        df = pd.read_csv(args.file)
        print(f"Đã đọc {len(df)} dòng từ file CSV.")
        
        # Kiểm tra xem dữ liệu có được đọc đúng không
        if len(df) == 0:
            print("File CSV không có dữ liệu!")
            sys.exit(1)
            
        # Tạo kết nối trực tiếp để kiểm tra một cách toàn diện
        # Sử dụng psycopg2 thay vì SQLAlchemy
        print(f"Đang kết nối trực tiếp đến PostgreSQL để kiểm tra thêm...")
        conn = psycopg2.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            database="postgres"
        )
        
        # Thử tạo database nếu chưa tồn tại
        conn.autocommit = True
        cursor = conn.cursor()
        try:
            cursor.execute(f"CREATE DATABASE {args.database}")
            print(f"Đã tạo database '{args.database}'")
        except psycopg2.errors.DuplicateDatabase:
            print(f"Database '{args.database}' đã tồn tại")
        
        conn.close()
        
        # Tiếp tục với kết nối đến database đích
        conn_target = psycopg2.connect(
            user=db_user,
            password=db_password, 
            host=db_host,
            port=db_port,
            database=args.database
        )
        
        # Thử tạo bảng nếu cần thiết
        conn_target.autocommit = True
        cursor = conn_target.cursor()
        if args.mode == "replace":
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {args.table}")
                print(f"Đã xóa bảng '{args.table}' cũ")
            except Exception as e:
                print(f"Không thể xóa bảng cũ: {e}")
        
        # Đóng kết nối psycopg2
        conn_target.close()
        
        # Tiếp tục với SQLAlchemy để load dữ liệu
        print(f"Đang load dữ liệu vào bảng {args.table}...")
        
        # Tạo kết nối SQLAlchemy trực tiếp với thông số đã hoạt động
        conn_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{args.database}"
        engine = create_engine(conn_string)
        print('Đang kết nối đến PostgreSQL...")')
        # Lưu dữ liệu
        df.to_sql(args.table, engine, if_exists=args.mode, index=False)
        print(f"Đã load thành công {len(df)} dòng vào {args.database}.{args.table}")
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        print("Đang chuyển sang giải pháp sử dụng SQLite...")
        
        # # Phương án dự phòng: Lưu vào SQLite
        # try:
        #     # Tạo kết nối SQLite
        #     sqlite_file = f"{args.database}.db"
        #     print(f"Đang lưu dữ liệu vào SQLite: {sqlite_file}")
            
        #     # Đọc dữ liệu nếu chưa đọc
        #     if 'df' not in locals():
        #         df = pd.read_csv(args.file)
            
        #     # Lưu vào SQLite
        #     import sqlite3
        #     conn = sqlite3.connect(sqlite_file)
        #     df.to_sql(args.table, conn, if_exists=args.mode, index=False)
        #     conn.close()
            
        #     print(f"Đã lưu thành công {len(df)} dòng vào SQLite: {sqlite_file}")
        #     print(f"   Bạn có thể truy vấn dữ liệu bằng lệnh: sqlite3 {sqlite_file}")
        # except Exception as sqlite_error:
        #     print(f"Lỗi khi lưu vào SQLite: {str(sqlite_error)}")
        #     sys.exit(1)

if __name__ == "__main__":
    main()