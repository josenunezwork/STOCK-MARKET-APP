from flask import Flask, render_template, request, jsonify
from scraper import download_stock_data
from databasemanager import create_tables, insert_stock_data, update_stock_data, retrieve_stock_data
from stock_ml import preprocess_data, load_model
import torch
import numpy as np
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/stock_data', methods=['GET', 'POST'])
def stock_data():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        # Download stock data
        stock_df = download_stock_data(ticker, start_date, end_date)
        
        if stock_df is not None:
            # Insert or update stock data in the database
            insert_stock_data(stock_df, ticker)
            update_stock_data(stock_df, ticker)
            
            return jsonify({'message': 'Stock data downloaded and stored successfully'})
        else:
            return jsonify({'error': 'Failed to download stock data'})
    
    return render_template('stock_data.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        ticker = request.form['ticker']
        
        # Retrieve stock data from the database
        stock_data = retrieve_stock_data(ticker)
        
        if stock_data is not None:
            # Preprocess the stock data
            X, y, scaler = preprocess_data(stock_data)
            
            # Reshape the data to match the input shape of the model
            X = X.reshape((X.shape[0], X.shape[1], 1, 1))
            
            # Load the trained model
            model = load_model('stock_prediction_model.pth')
            model.eval()
            
            # Make predictions
            with torch.no_grad():
                X_tensor = torch.tensor(X).float()
                predictions = model(X_tensor).numpy().flatten()
                
            
            # Get the actual closing prices for the corresponding timesteps
            actual_prices = y
            # Inverse transform predictions and actual prices
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

            # Generate the dates for the x-axis labels
            # Generate the dates for the x-axis labels
            dates = stock_data['date'].tolist()[60:]
            
            return jsonify({
                'predictions': predictions.tolist(),
                'actual_prices': actual_prices.tolist(),
                'dates': dates
            })
        else:
            return jsonify({'error': 'No stock data found for the specified ticker'})
    
    return render_template('predict.html')

if __name__ == '__main__':
    # Create the database tables
    create_tables()
    
    # Run the Flask app
    app.run(debug=True)