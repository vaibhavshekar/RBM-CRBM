import torch
from torchvision import datasets, transforms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_image_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, train_data

def load_tabular_data(csv_path, batch_size):
    df = pd.read_csv(csv_path)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    dataset = torch.utils.data.TensorDataset(torch.tensor(features).float(), torch.tensor(labels).long())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, features, labels
