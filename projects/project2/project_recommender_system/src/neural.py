import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from parsers import load_data

import torch.optim as optim


def main():
    train, test, transformation_user, transformation_item = load_data(0)
    nz_items, nz_users = train.nonzero()
    ratings = train[nz_items, nz_users].toarray()

    x_train = torch.from_numpy(np.stack((nz_items, nz_users)))
    y_train = torch.from_numpy(ratings)

    ds = TensorDataset(np.stack((nz_items, nz_users)), ratings)
    dl = DataLoader(ds)

    net = Net()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dl, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # clear the gradients of the variables
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


# define Neural Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5)  # 1 single channel per input, 6 channels per output and 5x5 filters
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 4)
        self.fc3 = nn.Linear(4, 5)  # the final 5 is because there are 5 possibilities (1-5)

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


# net = Net()

if __name__ == '__main__':
    main()
