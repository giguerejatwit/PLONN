import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from data import matchupA, matchupB, matchupC, matchupD, X_train_normal, X_test_normal, y_train, y_test, predict_data_normal

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        # x = self.sigmoid(x)
        return x.squeeze()



# Convert data into tensors

# features = MatchA[['W', 'L', 'GA/G', 'GF/G', 'SV%']].values
# MatchA.reset
def flatten(matchup):
    team1_data = matchup.iloc[0].values
    team2_data = matchup.iloc[1].values

    return pd.Series(list(team1_data) + list(team2_data))

flattenedA = flatten(matchupA)
flattenedB = flatten(matchupB)
flattenedC = flatten(matchupC)
flattenedD = flatten(matchupD)
combined_goals_A = flattenedA[3] + flattenedA[8]
combined_goals_B = flattenedB[3] + flattenedB[8]
combined_goals_C = flattenedC[3] + flattenedC[8]
combined_goals_D = flattenedD[3] + flattenedD[8]


all_data = pd.concat([flattenedA, flattenedB, flattenedC, flattenedD], axis=1).T
# Normalize data
normalized_data = StandardScaler().fit_transform(all_data)

# 
X_train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_normal, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

predict_tensor = torch.tensor(predict_data_normal, dtype=torch.float32)

labels = torch.tensor([combined_goals_A, combined_goals_B, combined_goals_C, combined_goals_D], dtype=torch.float32) 
features = torch.tensor(normalized_data, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
predict_dataset = TensorDataset(predict_tensor, labels)
dataset = TensorDataset(features, labels)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

predict_loader = DataLoader(dataset=predict_dataset, batch_size = 2, shuffle = False)


# model data

model = NeuralNetwork(input_size=X_train_normal.shape[1], hidden_size=10).to(device)

# Binary Cross-Entropy Loss
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 50

for epoch in range(epochs):
    # Training
    model.train()
    for batch_data, batch_labels in train_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        outputs = model(batch_data).squeeze()
        loss = loss_fn(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data, batch_labels in predict_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data).squeeze()
            loss = loss_fn(outputs, batch_labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(predict_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_loss:.4f}")


for epoch in range(epochs):
     for batch_data, batch_labels in data_loader:
         batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
         outputs = model(batch_data).squeeze()
         loss = loss_fn(outputs, batch_labels)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
   
     print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print('\n')


with torch.no_grad():
    all_predictions1 = model(features.to(device))
    all_predictions2 = model(predict_tensor.to(device))

    # convert probabilities to labels
    pred_label1 = (all_predictions1.squeeze() > 6.5).float()
    pred_label2 = (all_predictions2.squeeze() > 6.5).float()

print('Results from historic data:', pred_label1)
print(all_predictions1.squeeze(), '\n')

print('Results from historic data', pred_label2)
print(all_predictions2.squeeze(), '\n')