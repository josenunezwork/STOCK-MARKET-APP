# database.py
from sqlalchemy import create_engine, Column, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import insert
import pandas as pd
# Create database engine
engine = create_engine('sqlite:///stocks.db')
Base = declarative_base()
# Create a session factory
Session = sessionmaker(bind=engine)
# Define the Stock model
class Stock(Base):
    __tablename__ = 'stocks'
    date = Column(Date, primary_key=True)
    ticker = Column(String, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)
# Create the database tables
def create_tables():
    Base.metadata.create_all(engine)
def insert_stock_data(df, ticker):
    """
    Insert stock data into the database.
    Args:
        df (pandas.DataFrame): DataFrame containing the stock data.
        ticker (str): Stock ticker symbol.
    """
    # Create a new session
    session = Session()
    session = Session()
    try:
        session.bulk_insert_mappings(Stock, df.to_dict(orient='records'))
        session.commit()
        print(f"Stock data for {ticker} inserted into the database.")
    except Exception as e:
        pass
    try:
        # Iterate over the rows of the DataFrame
        for _, row in df.iterrows():
            # Create a new Stock object
            stock = Stock(
                date=row['date'],
                ticker=ticker,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                adj_close=row['adj_close'],
                volume=row['volume']
            )
            # Add the Stock object to the session
            session.add(stock)
        
        # Commit the changes to the database
        session.commit()
        print(f"Stock data for {ticker} inserted into the database.")
    except Exception as e:
        # Rollback the transaction in case of an error
        session.rollback()
        print(f"An error occurred while inserting data for {ticker}: {str(e)}")
    finally:
        # Close the session
        session.close()

def update_stock_data(df, ticker):
    """
    Update stock data in the database.
    Args:
        df:DataFrame containing the stock data.
        ticker (str): Stock ticker symbol.
    """
    session = Session()
    try:
        for _, row in df.iterrows():
            stock = session.query(Stock).filter_by(date=row['date'], ticker=ticker).first()
            # Check if the stock data exists in the database
            if stock:
                stock.open = row['open']
                stock.high = row['high']
                stock.low = row['low']
                stock.close = row['close']
                stock.adj_close = row['adj_close']
                stock.volume = row['volume']
            else:
                stock = Stock(
                    date=row['date'],
                    ticker=ticker,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    adj_close=row['adj_close'],
                    volume=row['volume']
                )
                session.add(stock)
        session.commit()
        print(f"Stock data for {ticker} updated in the database.")
    except Exception as e:
        # Rollback the transaction in case of an error
        session.rollback()
        print(f"An error occurred while updating data for {ticker}: {str(e)}")
    finally:
        session.close()
def retrieve_stock_data(ticker):
    """
    Retrieve stock data from the database.
    Args:
        ticker (str): Stock ticker symbol.
    Returns:
        DataFrame with stock data.
    """
    # Create a new session
    session = Session()
    
    try:
        # Retrieve all stock data for the given ticker
        stocks = session.query(Stock).filter_by(ticker=ticker).all()
        
        if stocks:
            # Convert the retrieved data into a DataFrame
            data = []
            for stock in stocks:
                data.append({
                    'date': stock.date,
                    'open': stock.open,
                    'high': stock.high,
                    'low': stock.low,
                    'close': stock.close,
                    'adj_close': stock.adj_close,
                    'volume': stock.volume
                })
            
            df = pd.DataFrame(data)
            return df
        else:
            print(f"No stock data found for {ticker} in the database.")
            return None
    except Exception as e:
        print(f"An error occurred while retrieving data for {ticker}: {str(e)}")
        return None
    finally:
        # Close the session
        session.close()
# TEST
if __name__ == '__main__':
    create_tables()
    ticker = 'AMZN'
    start_date = '2012-01-01'
    end_date = '2023-06-09'
    from scraper import download_stock_data
    stock_data = download_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        insert_stock_data(stock_data, ticker)
        update_stock_data(stock_data, ticker)
        retrieved_data = retrieve_stock_data(ticker)
        if retrieved_data is not None:
            print("Retrieved stock data:")
            print(retrieved_data)
