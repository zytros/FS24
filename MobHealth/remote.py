# Import necessary libraries
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sktime.regression.deep_learning.resnet import ResNetRegressor
from sktime.regression.deep_learning.cnn import CNNRegressor
from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def passband_heart(ppg_signal, fs):
    # Bandpass filter to isolate heart rate component
    # : highcut to 160bpm = 2.67Hz
    lowcut = 0.67 / (0.5 * fs)  # 40 bpm
    highcut = 2.67 / (0.5 * fs)  # 240 bpm
    b, a = butter(1, [lowcut, highcut], btype='band')
    filtered_signal = lfilter(b, a, ppg_signal)
    return filtered_signal

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
def prepare_data(phase,y):
    INTERVAL_LENGTH = 128*30
    X = []
    for i in range(len(y)):
        X.append(phase[i*INTERVAL_LENGTH:(i+1)*INTERVAL_LENGTH])
    return np.array(X)

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_dataloaders_from_arrays(X_train, y_train, X_test, y_test, batch_size=16):
    # Convert arrays to PyTorch tensors
    X_train_torch = torch.tensor(X_train)
    y_train_torch = torch.tensor(y_train)
    X_test_torch = torch.tensor(X_test)
    y_test_torch = torch.tensor(y_test)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, num_epochs=10, learning_rate=0.001):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    return model

def test_model(model, dl_test, device='cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dl_test:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dl_test.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(dl_test.dataset),
        100. * correct / len(dl_test.dataset)))
    
def predict(model, data):
    # Convert the data to a PyTorch tensor and add an extra dimension
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = data_tensor.to(device)
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Make the prediction
    with torch.no_grad():
        output = model(data_tensor)

    # Convert the output tensor to a numpy array and return it
    return output.cpu().numpy()

