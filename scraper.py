# stock_scraper.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
def download_stock_data(ticker, start_date=None, end_date=None):
    """
    scrape stock data for a given ticker and date range using yfinance api
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data download (YYYY-MM-DD).
        end_date (str): End date for data download (YYYY-MM-DD).
    Returns:
        DataFrame containing scraped stock data.
    """
    # Check if start_date and end_date are provided
    if start_date is None:
        start_date = '2010-01-01'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        # Download stock data using yfinance api
        data = yf.download(ticker, start=start_date, end=end_date)
        # Check for missing data
        if data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}")
            return None
        # Reset the index to convert date column to a regular column
        data.reset_index(inplace=True)
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        data.rename(columns=column_mapping, inplace=True)
        # Check the last downloaded date
        last_date = data['date'].max()
        # Check for gaps between last downloaded date and current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        if last_date < current_date:
            # Download the missing data from the last downloaded date to the current date
            missing_data = yf.download(ticker, start=last_date, end=current_date)
            # Check if missing data is not empty
            if not missing_data.empty:
                # Reset the index of missing data
                missing_data.reset_index(inplace=True)
                missing_data.rename(columns=column_mapping, inplace=True)
                
                # Concatenate the missing data with the original data
                data = pd.concat([data, missing_data], ignore_index=True)
        return data
    except Exception as e:
        print(f"An error occurred while downloading data for {ticker}: {str(e)}")
        return None

def save_stock_data(df, ticker):
    """
    Save the downloaded stock data to a CSV file
    Args:
        df (pandas.DataFrame): DataFrame containing the stock data.
        ticker (str): Stock ticker symbol.
    """
    # Create a filename based on the ticker and current date
    current_date = datetime.now().strftime('%Y%m%d')
    filename = f"{ticker}_{current_date}.csv"
    try:
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        print(f"Stock data for {ticker} saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving data for {ticker}: {str(e)}")
# TEST
if __name__ == '__main__':
    ticker = 'AAPL'  
    start_date = '2022-01-01'  
    end_date = '2023-06-09'  
    # Download stock data
    stock_data = download_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        # Save the downloaded data to a CSV file
        save_stock_data(stock_data, ticker)