# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import pdb

# Set print options for numpy and torch
np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)

# Function to slice a dataset based on a percentage
def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    return dataset[:int(data_size * percentage / 100)]

# Determine if GPU (cuda) is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to create a dataset
# Function to create a dataset
def make_dataset(dataset, n_bus):
    # Initialize lists to store raw and normalized data
    x_raw, y_raw = [], []
    slack_encoded = []
    # Loop through the dataset
    for i in range(len(dataset)):
        x_raw_1 = []
        y_raw_1 = []
        for n in range(n_bus):
            # Extract data for each bus
            # Starting index is shifted by one to skip the first feature
            bus_features_start_index = n * 5 + 1
            bus_features_end_index = bus_features_start_index + 4
            slack_encode = n * 5 + 5
            # Append the next three features to the input
            x_raw_1.append(dataset[i, bus_features_start_index:bus_features_end_index])
            # Append the last two features to the output
            y_raw_1.append(dataset[i, bus_features_start_index + 2:bus_features_end_index])
            
        # Note that we've moved the append outside of the inner loop
        slack_encoded.append(slack_encode)
        x_raw.append(x_raw_1)
        y_raw.append(y_raw_1)

    # Convert lists to PyTorch tensors
    x_raw = torch.tensor(x_raw, dtype=torch.float)
    y_raw = torch.tensor(y_raw, dtype=torch.float)
    slack_encoded = torch.tensor(slack_encoded, dtype=torch.float)
    return x_raw, y_raw, slack_encoded


# Function to normalize the dataset
def normalize_dataset(x, y, n_bus):
    x_mean = torch.mean(x,0)
    y_mean = torch.mean(y,0)
    x_std = torch.std(x,0)
    y_std = torch.std(y,0)
    x_norm = (x-x_mean)/x_std
    y_norm = (y-y_mean)/y_std
    x_norm = torch.where(torch.isnan(x_norm), torch.zeros_like(x_norm), x_norm)
    y_norm = torch.where(torch.isnan(y_norm), torch.zeros_like(y_norm), y_norm)
    x_norm = torch.where(torch.isinf(x_norm), torch.zeros_like(x_norm), x_norm)
    y_norm = torch.where(torch.isinf(y_norm), torch.zeros_like(y_norm), y_norm)
    return x_norm, y_norm, x_mean, y_mean, x_std, y_std

# Function to denormalize the output
def denormalize_output(y_norm, y_mean, y_std):
    y = y_norm * y_std + y_mean
    return y

# Define NRMSE (Normalized Root Mean Square Error) function
def NRMSE(yhat, y):
    return torch.sqrt(torch.mean(((yhat - y) / torch.std(yhat, 0))**2))

# Define MSE (Mean Square Error) function
def MSE(yhat, y):
    return torch.mean((yhat - y)**2)

# Load dataset from Excel files
dataset1 = pd.read_excel("D:/University of Bremen/Master Project/Dataset for different bus systems/Data Generated_IEEE30.xlsx").values

# Set the percentage for training and validation data
train_percentage = 70
val_percentage = 30

# Slice the dataset into training and validation subsets
train_dataset = slice_dataset(dataset1, train_percentage)
val_dataset = slice_dataset(dataset1, val_percentage)

n_bus = 30  # Number of buses

# Extract raw data for training and validation
x_raw_train, y_raw_train, slack_train = make_dataset(train_dataset, n_bus)
x_raw_val, y_raw_val, slack_val = make_dataset(val_dataset, n_bus)

# Normalize the data
x_norm_train, y_norm_train, _, _, _, _ = normalize_dataset(x_raw_train, y_raw_train, n_bus)
x_norm_val, y_norm_val, x_val_mean, y_val_mean, x_val_std, y_val_std = normalize_dataset(x_raw_val, y_raw_val, n_bus)
x_train, y_train = x_norm_train, y_norm_train
x_val, y_val = x_norm_val, y_norm_val

# Define the edge index (graph structure)
edge_index = torch.tensor([
    [0, 0, 1, 2, 1, 1, 3, 4, 5, 5, 11, 11, 11, 13, 15, 14, 17, 18, 9, 9, 9, 9, 20, 14, 21, 22, 23, 24, 24, 26, 26, 28, 7, 5],
    [1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 13, 14, 15, 14, 16, 17, 18, 19, 19, 16, 20, 21, 21, 22, 23, 23, 24, 25, 26, 28, 29, 29, 27, 27]
], dtype=torch.long)


# Create a list of Data objects for training and validation
data_train_list, data_val_list = [], []
for i, _ in enumerate(x_train):
    data_train_list.append(Data(x=x_train[i], y=y_train[i], edge_index=edge_index,slack=slack_train[i]))
for i, _ in enumerate(x_val):
    data_val_list.append(Data(x=x_val[i], y=y_val[i], edge_index=edge_index, slack=slack_val))

# Define batch size
n_batch = 50

# Create DataLoader for training and validation data
train_loader = DataLoader(data_train_list, batch_size=n_batch)
val_loader = DataLoader(data_val_list, batch_size=n_batch)

# Define a custom GNN (Graph Neural Network) model
class My_GNN_GNN_NN(torch.nn.Module):
    def __init__(self, node_size, feat_in, feat_size1, feat_size2, hidden_size1, output_size, n_batch):
        super(My_GNN_GNN_NN, self).__init__()
        # Corrected the feature size assignments
        self.conv1 = GCNConv(feat_in, feat_size1)
        self.conv2 = GCNConv(feat_size1, feat_size2)
        self.lin1 = Linear(feat_size2, hidden_size1)
        self.lin2 = Linear(hidden_size1, output_size)
        self.n_batch = n_batch
        self.node_size=node_size
        self.feat_size2=feat_size2

    def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = torch.tanh(x)
            x = self.conv2(x, edge_index)
            x = torch.tanh(x)
            x = torch.flatten(x, start_dim=1)
            x = self.lin1(x)
            x = F.relu(x)
            x = self.lin2(x)
            x = x.view(-1, 1500, 2)
            return x
    # Function to save the model's weights
    def save_weights(self, model, name):
        torch.save(model, name)

# Define model hyperparameters
feat_in = 4  # The number of input features for each node
feat_size1 = 8
feat_size2 = 3000
hidden_size1 = 30
# Ensure n_bus is defined somewhere in your code
output_size = n_bus * 2
lr = 0.0001

# Create an instance of the custom GNN model
model = My_GNN_GNN_NN(n_bus, feat_in, feat_size1, feat_size2, hidden_size1, output_size, n_batch)

# Print model parameter names and sizes
for name, param in model.named_parameters():
    print(name)
    print(param.size())

# Calculate the total number of model parameters
param = sum(p.numel() for p in model.parameters() if p.requires_grad)
param  
# Define an Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Lists to store training and validation loss
train_loss_list, val_loss_list = [], []

count = 0
patience = 5
lossMin = 1e10

# Training loop
for epoch in range(2001):
    model.train()
    train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        y_train_prediction = model(batch)
        # print(y_train_prediction.shape)
        # print(batch.y.shape)
        loss = MSE(y_train_prediction, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs

    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)

    model.eval()
    val_loss = 0

    for batch in val_loader:
        y_val_prediction = model(batch)
        loss = MSE(y_val_prediction, batch.y)
        val_loss += loss.item() * batch.num_graphs

    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)

    # Early stopping
    if val_loss < lossMin:
        lossMin = val_loss
        count = 0
        best_epoch = epoch
        best_train_loss = train_loss
        best_val_loss = val_loss
        model.save_weights(model, "[PyG] [14 bus] Best_GNN_GNN_NN_model.pt")
    else:
        count += 1
        if count > patience:
            print("Early stop at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss, val_loss))
            print("Best val at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(best_epoch, best_train_loss, best_val_loss))
            break

    if train_loss <= 0:
        print("Min train loss at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss, val_loss))
        break

    if (epoch % 10) == 0:
        print('Epoch: {:d}    train loss: {:.7f}    val loss: {:.7f}'.format(epoch, train_loss, val_loss))

# Plot the training and validation loss
plt.plot(train_loss_list, label="Train loss")
plt.plot(val_loss_list, label="Validation loss")
plt.yscale('log')
plt.xlabel("# Epoch")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()

print('Last epoch: {:d}, train loss: {:.7f}, val loss: {:.7f}'.format(epoch, train_loss, val_loss))
print('Best epoch: {:d}, train loss: {:.7f}, val loss: {:.7f}'.format(best_epoch, best_train_loss, best_val_loss))
# End of the code
