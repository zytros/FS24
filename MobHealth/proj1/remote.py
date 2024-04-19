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

def prepare_full_data(ppg, accx, accy, accz):
    INTERVAL_LENGTH = 128*30
    # Normalize the PPG signal
    ppg_normalized = normalize_data(ppg)
    accx_normalized = normalize_data(accx)
    accy_normalized = normalize_data(accy)
    accz_normalized = normalize_data(accz)
    ds = []
    for i in range(len(ppg) // INTERVAL_LENGTH):
        ppg_interval = ppg_normalized[i*INTERVAL_LENGTH:(i+1)*INTERVAL_LENGTH]
        accx_interval = accx_normalized[i*INTERVAL_LENGTH:(i+1)*INTERVAL_LENGTH]
        accy_interval = accy_normalized[i*INTERVAL_LENGTH:(i+1)*INTERVAL_LENGTH]
        accz_interval = accz_normalized[i*INTERVAL_LENGTH:(i+1)*INTERVAL_LENGTH]
        ds.append([ppg_interval, accx_interval, accy_interval, accz_interval])
    

    # Combine the normalized signals into a single array
    data = np.array(ds)

    return data

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
    X_train_torch = torch.tensor(X_train.astype(np.float32))
    y_train_torch = torch.tensor(y_train.astype(np.float32))
    X_test_torch = torch.tensor(X_test.astype(np.float32))
    y_test_torch = torch.tensor(y_test.astype(np.float32))

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

sampling_rate = 128  # Hz

# Load data item containing the PPG, HR, and IMU signals from all phases
data = np.load('mhealth24_data_public.npy', allow_pickle=True).item() # now it is a dict

print('Keys for data:', data.keys())

# Example to extract the data from phase 0
# phase 0,3: wrist
# testing: phases 1,3,5
phase0_data = data['phase 0']
phase1_data = data['phase 1']
phase2_data = data['phase 2']
phase3_data = data['phase 3']
phase4_data = data['phase 4']
phase5_data = data['phase 5']
print('Keys for phase 0:', phase0_data.keys())

# Get the individual signals from phase 0
ppg_phase0 = phase0_data['PPG wrist']
ref_hr_phase0 = phase0_data['ground truth HR']  # only available for phase 0, 2, and 4 (training data)
IMU_X_phase0 = phase0_data['IMU X wrist']
IMU_Y_phase0 = phase0_data['IMU Y wrist']
IMU_Z_phase0 = phase0_data['IMU Z wrist']
ppg_phase1 = phase1_data['PPG head']
IMU_X_phase1 = phase1_data['IMU X head']
IMU_Y_phase1 = phase1_data['IMU Y head']
IMU_Z_phase1 = phase1_data['IMU Z head']
ppg_phase2 = phase2_data['PPG head']
ref_hr_phase2 = phase2_data['ground truth HR']
IMU_X_phase2 = phase2_data['IMU X head']
IMU_Y_phase2 = phase2_data['IMU Y head']
IMU_Z_phase2 = phase2_data['IMU Z head']
ppg_phase3 = phase3_data['PPG wrist']
IMU_X_phase3 = phase3_data['IMU X wrist']
IMU_Y_phase3 = phase3_data['IMU Y wrist']
IMU_Z_phase3 = phase3_data['IMU Z wrist']
ppg_phase4 = phase4_data['PPG head']
ref_hr_phase4 = phase4_data['ground truth HR']
IMU_X_phase4 = phase4_data['IMU X head']
IMU_Y_phase4 = phase4_data['IMU Y head']
IMU_Z_phase4 = phase4_data['IMU Z head']
ppg_phase5 = phase5_data['PPG head']
IMU_X_phase5 = phase5_data['IMU X head']
IMU_Y_phase5 = phase5_data['IMU Y head']
IMU_Z_phase5 = phase5_data['IMU Z head']

data_x = np.concatenate((ppg_phase2, ppg_phase4), axis=0)
y = np.concatenate((ref_hr_phase2, ref_hr_phase4), axis=0)
data_x.shape, y.shape

y = ref_hr_phase0

sig_flt = passband_heart(ppg_phase0, sampling_rate)
X = prepare_data(sig_flt,y)
X = normalize_data(X)
X_train, X_test, y_train, y_test = split_data(X, y)

D = prepare_full_data(sig_flt, IMU_X_phase0, IMU_Y_phase0, IMU_Z_phase0)
print(D.shape)
D_train, D_test, yD_train, yD_test = split_data(D, y)

# Function to print the mean and median absolute error between your predicted HR and the reference HR
# With this function, you can evaluate the resulting score that you would obtain on the public dataset
# with your predicted HR values on Kaggle
def print_score(pred_hr, ref_hr):
    err = np.abs(np.asarray(pred_hr) - np.asarray(ref_hr))
    print("Mean error: {:4.3f}, Median error {:4.3f}".format(np.mean(err), np.median(err)))
    s = 0.5 * np.mean(err) + 0.5 * np.median(err)
    print("Resulting score {:4.3f}".format(s))
    return s

# Example on how to use the print_score function with randomly generated HR values as the predictions
pred_hr_phase0 = list(np.random.randint(40, 180, len(ref_hr_phase0)))
_ = print_score(pred_hr_phase0, ref_hr_phase0)

def better_score(pred, ref, desc):
    # Calculate the score
    score = print_score(pred, ref)
    with open('losses.txt', 'a') as f:
        f.write("Score for " + desc + ": " + str(score) + "\n")
    return score


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 8, kernel_size=63, padding=1) #3840 -> 3588
        self.maxpool1 = nn.MaxPool1d(3) #3588 -> 1196
        self.conv2 = nn.Conv1d(8, 16, kernel_size=31, padding=1) #1196 -> 1132
        self.maxpool2 = nn.MaxPool1d(2) #1132 -> 566
        self.conv3 = nn.Conv1d(16, 32, kernel_size=15, padding=1) #566 -> 552
        self.avgpool1 = nn.AvgPool1d(3) #552 -> 185
        self.fc1 = nn.Linear(201*32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.do = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.maxpool1(torch.relu(self.conv1(x)))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = self.avgpool1(torch.relu(self.conv3(x)))
        x = x.view(-1, 201*32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.do(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
train_loader, test_loader = create_dataloaders_from_arrays(D_train, yD_train, D_test, yD_test)
train_loader.dataset[0][0].shape

conv_net = ConvNet()
#TODO: increase num_epochs
train_model(conv_net, train_loader, num_epochs=100000)

pred = predict(conv_net, D_test.astype(np.float32)[0])

better_score(pred, yD_test, 'ConvNet 1/2 s kernels 100000 epochs')

torch.save(conv_net.state_dict(), 'conv_net.pth')