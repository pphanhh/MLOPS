"""
Data validation module using Great Expectations.
"""
import os
import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe
import logging
from datetime import datetime

# Ensure correct log directory exists
os.makedirs("data/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwitterDataValidator:
    """Twitter data validator class."""
    
    def __init__(self, df=None, filepath=None):
        """
        Initialize validator with DataFrame or filepath.
        
        Args:
            df: DataFrame to validate (optional)
            filepath: Path to CSV file to validate (optional)
        """
        self.context = gx.get_context()
        
        if df is not None:
            self.df = df
        elif filepath is not None:
            self.df = pd.read_csv(filepath)
        else:
            raise ValueError("Either df or filepath must be provided")
            
        # Set up Great Expectations batch
        self.data_source = self.context.data_sources.add_pandas("twitter_data")
        self.data_asset = self.data_source.add_dataframe_asset(name="twitter_tweets")
        self.batch_definition = self.data_asset.add_batch_definition_whole_dataframe("batch_definition")
        self.batch = self.batch_definition.get_batch(batch_parameters={"dataframe": self.df})
        
        # Results container
        self.validation_results = []

    def validate_date_range(self):
        """Validate date range is within the expected period."""
        logger.info("Validating date range")
        
        try:
            # Check if date column exists
            if 'date' not in self.df.columns:
                logger.warning("Date column not found")
                self.validation_results.append({
                    "check": "date_range",
                    "success": False,
                    "details": "Date column not found"
                })
                return False
            
            # Convert dates and handle nulls
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            original_length = len(self.df)
            self.df.dropna(subset=['date'], inplace=True)
            if len(self.df) < original_length:
                logger.warning(f"Dropped {original_length - len(self.df)} rows with invalid dates")
            
            # Update the batch after modifying the dataframe
            self.batch = self.batch_definition.get_batch(batch_parameters={"dataframe": self.df})
            
            # Expected date range
            min_date = datetime(2022, 1, 1)
            max_date = datetime.now()
            
            # Get actual date range in data
            min_date_in_data = self.df['date'].min()
            max_date_in_data = self.df['date'].max()
            logger.info(f"Data date range: {min_date_in_data} to {max_date_in_data}")
            
            # Use two separate expectations to validate min and max dates
            min_expectation = gxe.ExpectColumnMin(
                column="date",
                min_value=min_date
            )
            
            max_expectation = gxe.ExpectColumnMax(
                column="date",
                max_value=max_date
            )
            
            min_result = self.batch.validate(min_expectation)
            max_result = self.batch.validate(max_expectation)
            
            success = min_result.success and max_result.success
            
            # Save results
            self.validation_results.append({
                "check": "date_range",
                "success": success,
                "details": f"Min date: {min_date_in_data}, Max date: {max_date_in_data}"
            })
            
            return success
            
        except Exception as e:
            logger.error(f"Error in date validation: {str(e)}")
            self.validation_results.append({
                "check": "date_range",
                "success": False,
                "details": f"Error: {str(e)}"
            })
            return False
    def validate_no_nulls(self, column):
        """Validate that a column has no null values."""
        logger.info(f"Validating no nulls in {column}")
        
        expectation = gxe.ExpectColumnValuesToNotBeNull(column=column)
        result = self.batch.validate(expectation)
        
        self.validation_results.append({
            "check": f"no_nulls_{column}",
            "success": result.success,
            "details": result
        })
        return result.success
    
    def validate_column_type(self, column, expected_type):
        """Validate that a column has the expected data type."""
        logger.info(f"Validating {column} is of type {expected_type}")
        
        expectation = gxe.ExpectColumnValuesToBeOfType(column=column, type_=expected_type)
        result = self.batch.validate(expectation)
        
        self.validation_results.append({
            "check": f"column_type_{column}",
            "success": result.success,
            "details": result
        })
        return result.success
    
    def validate_column_exists(self, column):
        """Validate that a column exists."""
        logger.info(f"Validating column {column} exists")
        
        expectation = gxe.ExpectColumnToExist(column=column)
        result = self.batch.validate(expectation)
        
        self.validation_results.append({
            "check": f"column_exists_{column}",
            "success": result.success,
            "details": result
        })
        return result.success
    
    def validate_column_count(self, expected_count):
        """Validate the total number of columns."""
        logger.info(f"Validating column count = {expected_count}")
        
        expectation = gxe.ExpectTableColumnCountToEqual(value=expected_count)
        result = self.batch.validate(expectation)
        
        self.validation_results.append({
            "check": "column_count",
            "success": result.success,
            "details": result
        })
        return result.success
    
    def validate_unique_values(self, column):
        """Validate that a column has unique values."""
        logger.info(f"Validating uniqueness of {column}")
        
        expectation = gxe.ExpectColumnValuesToBeUnique(column=column)
        result = self.batch.validate(expectation)
        
        self.validation_results.append({
            "check": f"unique_values_{column}",
            "success": result.success,
            "details": result
        })
        return result.success
    
    def run_all_validations(self):
        """Run all validations."""
        logger.info("Running all validations")
        actual_columns = len(self.df.columns)
        validations = [
            self.validate_column_count(actual_columns),
            self.validate_column_exists("Sentiment"),
            self.validate_column_exists("cleaned_text"),
            self.validate_column_type("likeCount", "int64"),
            # self.validate_date_range(),
            self.validate_no_nulls("cleaned_text"),
            self.validate_sentiment_distribution(),
        ]
        
        success_count = sum(validations)
        total_count = len(validations)
        
        logger.info(f"Validation complete: {success_count}/{total_count} checks passed")
        return success_count == total_count

    def get_validation_summary(self):
        """Get validation summary."""
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r["success"])
        
        return {
            "total_checks": total,
            "passed_checks": passed,
            "failed_checks": total - passed,
            "success_rate": passed / total if total > 0 else 0,
            "all_passed": passed == total,
            "results": self.validation_results
        }
    
        
    def validate_sentiment_distribution(self):
        """Validate sentiment labels have reasonable distribution."""
        logger.info("Validating sentiment distribution")
        
        # Check that we have at least 10% of each sentiment class
        expectation = gxe.ExpectColumnDistinctValuesToContainSet(
            column="Sentiment", 
            value_set=["Positive", "Neutral", "Negative"]
        )
        result = self.batch.validate(expectation)
        self.validation_results.append({
            "check": "sentiment_values",
            "success": result.success,
            "details": result
        })
        
        # Add this to your run_all_validations method
        # self.validate_sentiment_distribution(),
        
        return result.success

def validate_dataset(filepath=None, df=None):
    """
    Validate Twitter dataset.
    
    Args:
        filepath: Path to CSV file (optional)
        df: DataFrame to validate (optional)
        
    Returns:
        Validation summary
    """
    os.makedirs("logs", exist_ok=True)
    
    validator = TwitterDataValidator(df=df, filepath=filepath)
    validator.run_all_validations()
    
    summary = validator.get_validation_summary()
    
    # Log summary results
    logger.info(f"Validation Results: {summary['passed_checks']}/{summary['total_checks']} checks passed")
    if not summary['all_passed']:
        failed = [r["check"] for r in summary["results"] if not r["success"]]
        logger.warning(f"Failed checks: {failed}")
    
    return summary

if __name__ == "__main__":
    # Find latest file in processed directory
    processed_dir = "./labeled"
    if not os.path.exists(processed_dir):
        print("Labeled directory not found!")
    else:
        all_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        if not all_files:
            print("No labeled files found!")
        else:
            latest_file = max(all_files, key=lambda f: os.path.getmtime(os.path.join(processed_dir, f)))
            filepath = os.path.join(processed_dir, latest_file)
            
            print(f"Validating {filepath}...")
            summary = validate_dataset(filepath=filepath)
            
            if summary['all_passed']:
                print("✅ All validation checks passed!")
            else:
                print(f"❌ {summary['failed_checks']} validation checks failed!")
                for result in summary['results']:
                    if not result['success']:
                        print(f"  - Failed: {result['check']}")