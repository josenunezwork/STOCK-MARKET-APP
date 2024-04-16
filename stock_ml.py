# stock_ml_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import numpy as np


from torchvision.models import ResNet18_Weights
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def preprocess_data(df):
    """
    Preprocess the stock data for training.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the stock data.
        
    Returns:
        tuple: Tuple containing the preprocessed features (X) and target (y).
    """
    # Assuming the target variable is 'close' price
    target_col = 'close'
    
    # Create a new DataFrame with only the 'close' column
    data = df.filter([target_col])
    
    # Convert the DataFrame to a numpy array
    dataset = data.values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create the training dataset
    train_data = scaled_data[0:-1]  # Use all data except the last row as training data
    
    # Split the data into features (X) and target (y)
    X = []
    y = []
    
    for i in range(60, len(train_data)):
        X.append(train_data[i-60:i, 0])
        y.append(train_data[i, 0])
    
    # Convert X and y to numpy arrays
    X, y = np.array(X), np.array(y)
    
    # Reshape the data
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def split_data(X, y, test_size=0.2):
    """
    Split the preprocessed data into training and validation sets.
    
    Args:
        X (numpy.ndarray): Features.
        y (numpy.ndarray): Target.
        test_size (float): Proportion of the dataset to include in the validation set.
        
    Returns:
        tuple: Tuple containing the split data (X_train, X_val, y_train, y_val).
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_val, y_train, y_val
class StockPredictionModel(nn.Module):
    def __init__(self, input_shape):
        super(StockPredictionModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
    def forward(self, x):
        x = x.squeeze(2)  # Remove the second dimension (channel dimension)
        x = x.unsqueeze(1)  # Add a new channel dimension at index 1
        x = self.resnet(x)
        return x
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
    """
    Train the PyTorch model on the training data.
    
    Args:
        model (torch.nn.Module): PyTorch model to train.
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation target.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        
    Returns:
        tuple: Tuple containing the trained model and training loss history.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if device is not None:
        model = model.to(device)
    
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float().unsqueeze(1))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    train_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, train_loss_history

def save_model(model, filename='stock_prediction_model.pth'):
    """
    Save the trained PyTorch model to a file.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model.
        filename (str): Filename to save the model.
    """
    torch.save(model.state_dict(), filename)

def load_model(filename='stock_prediction_model.pth'):
    """
    Load a trained PyTorch model from a file.
    
    Args:
        filename (str): Filename of the saved model.
        
    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    input_shape = (60, 1, 1)
    model = StockPredictionModel(input_shape)
    model.load_state_dict(torch.load(filename))
    return model

# Example usage
if __name__ == '__main__':
    # Retrieve stock data from the database
    from databasemanager import retrieve_stock_data
    ticker = 'AAPL'  # Replace with the desired stock ticker
    stock_data = retrieve_stock_data(ticker)
    
    if stock_data is not None:
        # Preprocess the stock data
        X, y, scaler = preprocess_data(stock_data)
        
        # Reshape the data to match the input shape of the model
        X = X.reshape((X.shape[0], X.shape[1], 1, 1))
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = split_data(X, y)
        
        # Create the PyTorch model
        input_shape = (X.shape[1], 1, 1)
        model = StockPredictionModel(input_shape)
        
        # Train the model with a specified number of epochs
        epochs = 50  # Adjust the number of epochs as needed
        model, train_loss_history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
        
        # Save the trained model
        save_model(model)
        
        # Load the saved model
        loaded_model = load_model()
        
        # Evaluate the loaded model on the validation set
        loaded_model.eval()
        with torch.no_grad():
            val_outputs = loaded_model(torch.tensor(X_val).float())
            val_loss = nn.MSELoss()(val_outputs, torch.tensor(y_val).float()).item()
        print(f"Validation loss: {val_loss:.4f}")
