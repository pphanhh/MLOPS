import pandas as pd 
import numpy as np
import html
import re
import ast
import logging
import os
from datetime import datetime
# Create logs directory
os.makedirs("logs", exist_ok=True)

# Simple configuration with just console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean tweet text by removing URLs, special characters, and formatting.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text
    """
    if pd.isnull(text):
        return ""
    
    # Decode HTML entities (e.g., &amp; → &)
    text = html.unescape(text)
    
    # Remove all URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove @ mentions but keep the username (e.g., @user → user)
    text = re.sub(r"@(\w+)", r"\1", text)
    
    # Remove hashtags completely (e.g., #topic → "")
    text = re.sub(r"#\S+", "", text)
    
    # Remove square brackets but keep the content inside (e.g., [trump] → trump)
    text = re.sub(r"\[(\w+)\]", r"\1", text)
    
    # Keep only letters, numbers, basic punctuation (.,?!'), and spaces
    text = re.sub(r"[^A-Za-z0-9\s.,?!']", "", text)
    
    # Normalize whitespace (replace multiple spaces with one, and trim)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def clean_display_name(name):
    """
    Clean user display name by removing special characters.
    
    Args:
        name: User display name
        
    Returns:
        Cleaned name
    """
    if pd.isnull(name):
        return ""
    
    # Keep only letters, numbers, and spaces
    return re.sub(r'[^A-Za-z0-9\s]', '', name).strip()

def hashtags_to_comma_string(hashtags):
    """
    Convert a list of hashtags to a comma-separated string.
    
    Args:
        hashtags: List of hashtags or string representation of list
        
    Returns:
        Comma-separated string of hashtags without # symbol
    """
    if pd.isnull(hashtags) or hashtags == '':
        return ""
        
    # If it's a string representation of a list → convert it to an actual list
    if isinstance(hashtags, str):
        try:
            hashtags = ast.literal_eval(hashtags)
        except:
            return ""
            
    # Join hashtags into a single comma-separated string, removing the '#' symbol
    return ', '.join(tag.replace('#', '') for tag in hashtags)

def clean_query(text):
    """
    Clean search query text.
    
    Args:
        text: Search query
        
    Returns:
        Cleaned query
    """
    if pd.isnull(text):
        return ""
        
    text = re.sub(r"#", "", text)
    text = re.sub(r":", " ", text)
    return text.strip()

def preprocess_data(input_file, output_file=None):
    """
    Preprocess Twitter data.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output processed CSV file (optional)
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Starting preprocessing of {input_file}")
    
    # Read the data
    df = pd.read_csv(input_file)
    original_rows = len(df)
    logger.info(f"Loaded {original_rows} rows")
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {original_rows - len(df)} duplicate rows")
    
    # Drop unnecessary columns
    columns_to_drop = ['url', 'sourceLabel', 'retweetedTweet_id', 'quotedTweet_id', 'scraped_by_account']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    logger.info(f"Dropped columns: {columns_to_drop}")
    #Drop nan
    df.dropna(inplace=True)
    logger.info(f"Removed rows with NaN values, remaining rows: {len(df)}")
    # Filter English tweets only
    if 'lang' in df.columns:
        english_count_before = len(df)
        df = df[df['lang'] == 'en']
        df.drop(columns=['lang'], inplace=True)
        logger.info(f"Filtered to {len(df)} English tweets (removed {english_count_before - len(df)} non-English)")
    
    # Clean text
    logger.info("Cleaning text fields")
    df["cleaned_text"] = df["text"].apply(clean_text)
    
    # Remove rows with insufficient text
    short_text_count = len(df[df['cleaned_text'].str.split().str.len() < 4])
    df.drop(df[df['cleaned_text'].str.split().str.len() < 4].index, inplace=True)
    logger.info(f"Removed {short_text_count} rows with text shorter than 4 words")
    
    # Clean user display names
    df['user_displayname'] = df['user_displayname'].apply(clean_display_name)
    
    # Process hashtags
    if 'hashtags' in df.columns:
        df['hashtag_text'] = df['hashtags'].apply(hashtags_to_comma_string)
    
    # Clean search keywords
    if 'searched_keyword' in df.columns:
        df['searched_keyword'] = df['searched_keyword'].apply(clean_query)
    
    # Drop original text and hashtags columns
    df.drop(columns=['text', 'hashtags'], inplace=True, errors='ignore')
    if 'id' in df.columns:
        df['id'] = df['id'].astype(int)
    if 'user_id' in df.columns:
        df['user_id'] = df['user_id'].astype(int)

    # Process date and time
    if 'date' in df.columns:
        # Convert to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Add year and month-year columns
        df['year'] = df['date'].dt.year
        df['month_year'] = df['date'].dt.to_period('M').astype(str)
        
        # Sort by date
        df = df.sort_values(by='date').reset_index(drop=True)
    
    # Lowercase all column names before saving
    df.columns = [col.lower() for col in df.columns]
    
    # Save preprocessed data if output file is specified
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Preprocessed data saved to {output_file}")
    
    logger.info(f"Preprocessing complete. Final dataset has {len(df)} rows and {len(df.columns)} columns")
    return df
    
if __name__ == "__main__":
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./processed", exist_ok=True)
    
    # Default input is the latest file in raw directory
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not all_files:
        print("No input files found!")
    else:
        latest_file = max(all_files, key=lambda f: os.path.getmtime(os.path.join(input_dir, f)))
        input_file = os.path.join(input_dir, latest_file)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed", f"processed_twitter_{timestamp}.csv")
        
        # Run preprocessing
        df = preprocess_data(input_file, output_file)
        print(f"Processed {len(df)} rows. Output saved to {output_file}")