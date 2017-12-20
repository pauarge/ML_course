import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.parsers import load_data_2


class NNLinearRegression(nn.Module):
    def __init__(self):
        super(NNLinearRegression, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class Net(nn.Module):
    def __init__(self, i, j):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, i)
        self.fc2 = nn.Linear(i, j)
        self.fc3 = nn.Linear(j, 1)
        self.relu = nn.Sigmoid()

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(features, labels, model, lossfunc, optimizer, num_epoch):
    for epoch in range(num_epoch):
        # create torch variables corresponding to features and labels
        inputs = Variable(torch.from_numpy(features))
        targets = Variable(torch.from_numpy(labels))

        # compute model predictions and loss
        outputs = model(inputs)
        loss = lossfunc(outputs, targets)

        # do a backward pass and a gradient update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    elems_tr, ratings_tr, elems_ts, ratings_ts = load_data_2()

    lossfunc = nn.MSELoss()
    for i in range(100,101):
        for j in range(50,51):
            model = Net(i,j)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

            train(features=elems_tr,
                  labels=ratings_tr,
                  model=model,
                  lossfunc=lossfunc,
                  optimizer=optimizer,
                  num_epoch=300)

            inputs_ts = Variable(torch.from_numpy(elems_ts))
            target_ts = Variable(torch.from_numpy(ratings_ts))
            output = model(inputs_ts)
            loss = lossfunc(output, target_ts)
            print("2-->{}-->{}-->1".format(i,j))
            print("Error Test{}".format(loss))


if __name__ == '__main__':
    main()
