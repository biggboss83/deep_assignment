import os
import predictor as p
import simulator
import torch
import torch.nn as nn
import torch.nn.functional as functional

# Create an ANN.


class MyANN(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.out(x)

        return self.sigmoid(x)


dirname = os.path.dirname(__file__)
PATH1 = os.path.join(dirname, 'fancy_named_model_872')

# PATH = "fancy_named_model"
# print(PATH)

our_model = torch.load(PATH1)
# our_model.eval()

net = MyANN(14, 18, 1)
pd = p.Predictor('Cloudia', net, our_model)
simulator.simulate(2018, 0, pd)
