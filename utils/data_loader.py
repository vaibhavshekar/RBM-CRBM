import torch
from torchvision import datasets, transforms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_image_data(batch_size):
    """
    Load MNIST image data and return a DataLoader and dataset object.
    Args:
        batch_size (int): Batch size for DataLoader.
    Returns:
        train_loader (DataLoader): DataLoader for MNIST training data.
        train_data (Dataset): MNIST dataset object.
    """
    # Normalize MNIST images to range [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Download and load MNIST dataset
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, train_data


def load_tabular_data(csv_path, batch_size):
    """
    Load tabular data from a CSV file and return a DataLoader and dataset object.
    Args:
        csv_path (str): Path to the CSV file containing tabular data.
        batch_size (int): Batch size for DataLoader.
    Returns:
        loader (DataLoader): DataLoader for tabular data.
        dataset (TensorDataset): TensorDataset object containing features and labels.
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    # Load CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    features = df.iloc[:, :-1].values  # All columns except the last one
    labels = df.iloc[:, -1].values     # Last column
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    # Convert features and labels to PyTorch tensors
    features_tensor = torch.tensor(features).float()
    labels_tensor = torch.tensor(labels).long()
    
    # Create TensorDataset and DataLoader
    dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, dataset
