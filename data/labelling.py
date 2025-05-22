"""
Data labeling module for sentiment labeling using Gemini API.
"""
import os
import pandas as pd
import requests
import time
import re
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/labeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API configuration
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash-lite"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

def extract_sentiment(text):
    """
    Extract sentiment label from API response.
    
    Args:
        text: API response text
        
    Returns:
        Sentiment label: 'Positive', 'Neutral', 'Negative', or 'Unknown'
    """
    match = re.search(r"\b(Positive|Neutral|Negative)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "Unknown"

def classify_comment(comment):
    """
    Classify comment sentiment using Gemini API.
    
    Args:
        comment: Text to classify
        
    Returns:
        Sentiment classification
    """
    prompt = (
        f"Classify the overall sentiment of this comment as Positive, Neutral, or Negative only. "
        f"Do not explain. Just return one word only.\n\nComment: {comment}"
    )
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return extract_sentiment(raw_text)
    except Exception as e:
        logger.error(f"Classification error: {comment[:60]}... \n{e}")
        return "Error"

def label_dataset(input_file, output_file=None, text_column="cleaned_text", 
                 label_column="Sentiment", rate_limit_delay=2):
    """
    Label a dataset with sentiment classifications.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path (optional)
        text_column: Column containing text to classify
        label_column: Column to store classifications
        rate_limit_delay: Delay between API calls (seconds)
        
    Returns:
        DataFrame with sentiment labels
    """
    if not API_KEY:
        logger.error("API key not found. Please set GEMINI_API_KEY in .env file")
        raise ValueError("API key not found")

    logger.info(f"Starting sentiment labeling for {input_file}")
    
    # Read the data
    df = pd.read_csv(input_file)
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} rows from {input_file}")
    
    # Initialize sentiment column if it doesn't exist
    if label_column not in df.columns:
        df[label_column] = ""
    
    # Temporary output file for incremental saving
    temp_output_file = output_file if output_file else "data/labeled/labeled_temp.csv"
    
    # Track progress
    rows_processed = 0
    rows_labeled = 0
    
    # Process each row
    start_time = time.time()
    try:
        for i in range(len(df)):
            # Skip already labeled rows
            if df.at[i, label_column] not in ["", None, "Error", "Unknown"]:
                rows_processed += 1
                continue
                
            # Log progress every 10 rows
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_rows - i) / rate if rate > 0 else 0
                logger.info(f"Progress: {i}/{total_rows} rows ({i/total_rows:.1%}) - ETA: {eta/60:.1f} mins")
            
            # Get text to classify
            text = df.at[i, text_column]
            if pd.isna(text) or text == "":
                df.at[i, label_column] = "Unknown"
                rows_processed += 1
                continue
            
            # Classify text
            logger.debug(f"Classifying row {i+1}/{total_rows}")
            sentiment = classify_comment(text)
            df.at[i, label_column] = sentiment
            
            # Save progress incrementally
            if i % 10 == 0 or i == len(df) - 1:
                df.to_csv(temp_output_file, index=False)
                
            rows_processed += 1
            if sentiment not in ["Error", "Unknown"]:
                rows_labeled += 1
            
            # Apply rate limiting
            time.sleep(rate_limit_delay)
    
    except KeyboardInterrupt:
        logger.warning("Labeling interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during labeling: {e}")
    
    finally:
        # Final save
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Labeled data saved to {output_file}")
        
        # Log summary
        elapsed = time.time() - start_time
        logger.info(f"Labeling complete: {rows_labeled} rows labeled out of {rows_processed} processed")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        
        # Return the dataframe with labels
        return df

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("./labeled", exist_ok=True)
    
    # Find latest file in processed directory
    processed_dir = "./processed"
    if not os.path.exists(processed_dir):
        print("Processed directory not found!")
    else:
        all_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        if not all_files:
            print("No processed files found!")
        else:
            latest_file = max(all_files, key=lambda f: os.path.getmtime(os.path.join(processed_dir, f)))
            input_file = os.path.join(processed_dir, latest_file)
            
            # Generate output filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"./labeled/labeled_twitter_{timestamp}.csv"
            
            # Run labeling
            print(f"Labeling {input_file}...")
            label_dataset(input_file, output_file)
            print(f"Labeled data saved to {output_file}")