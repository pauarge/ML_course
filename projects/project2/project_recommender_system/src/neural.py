import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from parsers import load_data, load_data_2
import matplotlib.pyplot as plt
import torch.optim as optim


class NNLinearRegression(nn.Module):
    def __init__(self):
        super(NNLinearRegression, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(2,1)
        self.conv1 = nn.Conv2d(2, 2, 2)  # 1 single channel per input, 6 channels per output and 5x5 filters
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 1)  # the final 5 is because there are 5 possibilities (1-5)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(features, labels, model, lossfunc, optimizer, num_epoch):
    for epoch in range(num_epoch):
        # TODO: Step 1 - create torch variables corresponding to features and labels
        inputs = Variable(torch.from_numpy(features))
        targets = Variable(torch.from_numpy(labels))

        # TODO: Step 2 - compute model predictions and loss
        outputs = model(inputs)
        loss = lossfunc(outputs, targets)

        # TODO: Step 3 - do a backward pass and a gradient update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Loss: %.8f' % (epoch + 1, num_epoch, np.math.sqrt(loss.data[0])))


def main():
    elems_tr, ratings_tr, elems_ts, ratings_ts = load_data_2()

    lossfunc = nn.MSELoss()
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    train(features=elems_tr,
          labels=ratings_tr,
          model=model,
          lossfunc=lossfunc,
          optimizer=optimizer,
          num_epoch=250)

    inputs_ts = Variable(torch.from_numpy(elems_ts))
    target_ts = Variable(torch.from_numpy(ratings_ts))
    output = model(inputs_ts)
    loss = lossfunc(output, target_ts)
    print("Error Test{}".format(loss))


if __name__ == '__main__':
    main()
