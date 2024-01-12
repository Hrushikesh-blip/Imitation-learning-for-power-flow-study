# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import pandapower.networks as pn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader
import numpy as np
import pdb # Python debugger
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import ARMAConv
from torch_geometric.data import Data
# Function to construct the edge_index tensor for a power grid network.
def edgeindex(net,z):
    x=[]
    y=[]
    pdb.set_trace  # Insert a debugger breakpoint
    for i in range(z):
        # Extracts the 'from' and 'to' bus indices from each line in the network
        x.append(net.line.from_bus[i])
        y.append(net.line.to_bus[i])
    # Creates the edge_index tensor required for PyTorch Geometric graphs
    edge_index = torch.tensor([x,y], dtype=torch.long)
    return edge_index
# These lines create edge_index for three different power grid networks
grid1_edgeindex = edgeindex(pn.case89pegase(),159)
grid2_edgeindex = edgeindex(pn.case39(),33)
grid3_edgeindex = edgeindex(pn.case_ieee30(),33)
# Set printing options for numpy and torch to have better formatted outputs
np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5, sci_mode=False)
# Function to slice a dataset to a specified percentage size
def slice_dataset(dataset, percentage):
    data_size = len(dataset)
    return dataset[:int(data_size*percentage/100)]
# Determine if CUDA is available and set the default device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Function to prepare the dataset for training/validation
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
    y = y_norm*y_std+y_mean
    return y
def MSE(yhat,y):
    return torch.mean((yhat-y)**2)
# This function splits the dataset into training and validation sets, processes, and normalizes them.
def dataset(x):
    dataset = pd.read_excel("D:/University of Bremen/Master Project/Dataset for different bus systems"+x).values
    return dataset
train_percentage = 70
val_percentage = 30
dataset1 = dataset("/Data Generated_89pegase.xlsx")
dataset3 = dataset("/Data Generated_39.xlsx")
dataset5 = dataset("/Data Generated_IEEE30.xlsx")
# This function splits the dataset into training and validation sets, processes, and normalizes them.
def datalist(dataset1, n_bus, train_percentage, val_percentage):
    train_dataset = slice_dataset(dataset1, train_percentage)
    val_dataset = slice_dataset(dataset1, val_percentage)
    x_raw_train, y_raw_train, slack_train = make_dataset(train_dataset, n_bus)
    x_raw_val, y_raw_val, slack_val = make_dataset(val_dataset, n_bus)
    x_norm_train, y_norm_train, _, _, _, _ = normalize_dataset(x_raw_train, y_raw_train, n_bus)
    x_norm_val, y_norm_val, x_val_mean, y_val_mean, x_val_std, y_val_std = normalize_dataset(x_raw_val, y_raw_val, n_bus)
    x_train, y_train = x_norm_train, y_norm_train
    x_val, y_val = x_norm_val, y_norm_val
    return x_train,y_train, x_val, y_val,  x_val_mean, y_val_mean, x_val_std, y_val_std, x_norm_train, y_norm_train, x_raw_train, y_raw_train, x_raw_val, y_raw_val, x_norm_val, y_norm_val, slack_train, slack_val
x_train,y_train, x_val, y_val, x_val_mean, y_val_mean, x_val_std, y_val_std, x_norm_train, y_norm_train, x_raw_train, y_raw_train, x_raw_val, y_raw_val, x_norm_val, y_norm_val, slack_train, slack_val=datalist(dataset1,89, 70, 30)
x_train1,y_train1, x_val1, y_val1, x_val_mean1, y_val_mean1, x_val_std1, y_val_std1, x_norm_train1, y_norm_train1, x_raw_train1, y_raw_train1, x_raw_val1, y_raw_val1, x_norm_val1, y_norm_val1, slack_train1, slack_val1=datalist(dataset3,39, 70, 30)
x_train2,y_train2, x_val2, y_val2, x_val_mean2, y_val_mean2, x_val_std2, y_val_std2, x_norm_train2, y_norm_train2, x_raw_train2, y_raw_train2, x_raw_val2, y_raw_val2, x_norm_val2, y_norm_val2, slack_train2, slack_val2=datalist(dataset5,30, 70, 30)
data_train_list, data_val_list = [], []
for i,_ in enumerate(x_train):
    data_train_list.append(Data(x=x_train[i], y=y_train[i], edge_index=grid1_edgeindex,slack=slack_train[i]))
for i,_ in enumerate(x_train1):
    data_train_list.append(Data(x=x_train1[i], y=y_train1[i], edge_index=grid2_edgeindex,slack=slack_train1[i]))
for i,_ in enumerate(x_train2):
    data_train_list.append(Data(x=x_train2[i], y=y_train2[i], edge_index=grid3_edgeindex,slack=slack_train2[i])) 
for i,_ in enumerate(x_val):
    data_val_list.append(Data(x=x_val[i], y=y_val[i], edge_index=grid1_edgeindex,slack=slack_train[i]))
for i,_ in enumerate(x_val1):
    data_val_list.append(Data(x=x_val1[i], y=y_val1[i], edge_index=grid2_edgeindex,slack=slack_train1[i]))
for i,_ in enumerate(x_val2):
    data_val_list.append(Data(x=x_val2[i], y=y_val2[i], edge_index=grid3_edgeindex,slack=slack_train2[i]))
n_batch = 1
train_loader = DataLoader(data_train_list, batch_size=n_batch, shuffle=False)
val_loader = DataLoader(data_val_list, batch_size=n_batch, shuffle=False)
class My_GNN_GNN_NN(nn.Module):
    def __init__(self):
        super(My_GNN_GNN_NN, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 128)
        self.arma1 = ARMAConv(128, 32, num_stacks=4)
        self.arma2 = ARMAConv(32, 32, num_stacks=4)
        self.arma3 = ARMAConv(32, 32, num_stacks=4)
        self.arma4 = ARMAConv(32, 32, num_stacks=4)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.arma1(x, edge_index))
        x = F.relu(self.arma2(x, edge_index))
        x = F.relu(self.arma3(x, edge_index))
        x = F.relu(self.arma4(x, edge_index))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    def save_weights(self, model, name):
        torch.save(model, name)
lr = 0.0001
model = My_GNN_GNN_NN()
for name, param in model.named_parameters():
  print(name)
  print(param.size())
param = sum(p.numel() for p in model.parameters() if p.requires_grad)
param
model = My_GNN_GNN_NN()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_loss_list, val_loss_list = [], []
count=0
patience=3
lossMin = 1e10
for epoch in range(2001):

    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y_train_prediction = model(batch)
        loss = MSE(y_train_prediction, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    
    model.eval()
    val_loss=0
    for batch in val_loader:
        y_val_prediction = model(batch)
        loss = MSE(y_val_prediction, batch.y)
        val_loss += loss.item() * batch.num_graphs
    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)

    #early stopping
    if (val_loss < lossMin):
        lossMin = val_loss
        count = 0
        best_epoch = epoch
        best_train_loss = train_loss
        best_val_loss = val_loss
        model.save_weights(model, "[PyG] [14 bus] Best_GNN_GNN_NN_model.pt")
    else:
        count+=1
        if(count>patience):
            print("early stop at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss, val_loss))
            print("best val at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(best_epoch, best_train_loss, best_val_loss))
            break
    
    if (train_loss <= 0):
        print("min train loss at epoch {:d}    train loss: {:.7f}    val loss: {:.7f}".format(epoch, train_loss, val_loss))
        break

    if (epoch % 10) == 0:
        print('epoch: {:d}    train loss: {:.7f}    val loss: {:.7f}'.format(epoch, train_loss, val_loss))      # DANACH ZURUECK ZU val_loss
plt.title('GNN NN on power flow dataset')
plt.plot(train_loss_list, label="train loss")
plt.plot(val_loss_list, label="val loss")
plt.yscale('log')
plt.xlabel("# Epoch")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.show()
print('last epoch: {:d}, train loss: {:.7f}, val loss: {:.7f}'.format(epoch, train_loss, val_loss))
print('best epoch: {:d}, train loss: {:.7f}, val loss: {:.7f}'.format(best_epoch, best_train_loss, best_val_loss))
best_model = torch.load("[PyG] [14 bus] Best_GNN_GNN_NN_model.pt")
best_model.eval()
test_loss_list = []
n_bus=57
dataset = pd.read_excel('D:/University of Bremen/Master Project/Dataset for different bus systems/Data Generated_57.xlsx').values
test_percentage = 100
test_dataset = slice_dataset(dataset, test_percentage)
x_raw_test, y_raw_test = make_dataset(test_dataset, n_bus)
x_norm_test, y_norm_test, _, _, _, _ = normalize_dataset(x_raw_test, y_raw_test, n_bus)
    
x_test, y_test = x_norm_test, y_norm_test
grid4_edgeindex = torch.tensor([[ 0,  1,  2,  3,  3,  5,  5,  7,  8,  8,  8,  8, 12, 12,  0,  0,  0,  2,
          4,  6,  9, 10, 11, 11, 11, 13, 17, 18, 20, 21, 22, 25, 26, 27, 24, 29,
         30, 31, 33, 34, 35, 36, 36, 35, 21, 40, 40, 37, 45, 46, 47, 48, 49, 28,
         51, 52, 53, 43, 55, 55, 56, 37, 37],
        [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 14, 15, 16, 14,
          5,  7, 11, 12, 12, 15, 16, 14, 18, 19, 21, 22, 23, 26, 27, 28, 29, 30,
         31, 32, 34, 35, 36, 37, 38, 39, 37, 41, 42, 43, 46, 47, 48, 49, 50, 51,
         52, 53, 54, 44, 40, 41, 55, 48, 47]])
data_test_list = []
for j,_ in enumerate(x_test):
    data_test_list.append(Data(x=x_test[j], y=y_test[j], edge_index=grid4_edgeindex))

test_loader = DataLoader(data_test_list, batch_size=1, shuffle=True)
    
print('dataset {:d}'.format(i+1))
    
test_loss = 0
for batch in test_loader:
    y_test_prediction = best_model(batch)
    loss = MSE(y_test_prediction, batch.y)
    test_loss += loss.item() * batch.num_graphs
test_loss /= len(test_loader.dataset)
print(test_loss)
