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

        if epoch % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epoch, loss.data[0]))


def main():
    elems_tr, ratings_tr, elems_ts, ratings_te = load_data_2()

    lossfunct = nn.MSELoss()
    model = NNLinearRegression()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train(features=elems_tr,
          labels=ratings_tr,
          model=model,
          lossfunc=lossfunct,
          optimizer=optimizer,
          num_epoch=100)

    # x_train = torch.from_numpy(x)
    # y_train = torch.from_numpy(ratings.T)

    # ds = TensorDataset(x_train, y_train)
    # dl = DataLoader(ds)


#     net = Net()
#     print(net)
# 
#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 
#     for epoch in range(2):  # loop over the dataset multiple times
# 
#         running_loss = 0.0
#         for i, data in enumerate(dl, 0):
#             # get the inputs
#             inputs, labels = data
# 
#             # wrap them in Variable
#             inputs, labels = Variable(inputs), Variable(labels)
# 
#             # clear the gradients of the variables
#             optimizer.zero_grad()
# 
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
# 
#             # print statistics
#             running_loss += loss.data[0]
#             if i % 2000 == 1999:  # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
# 
#     print('Finished Training')
# 
# 
# # define Neural Net
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(2,5)
#         # self.conv1 = nn.Conv2d(1, 6, 5)  # 1 single channel per input, 6 channels per output and 5x5 filters
#         # #self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(2, 2)
#         # self.fc2 = nn.Linear(2, 4)
#         # self.fc3 = nn.Linear(4, 5)  # the final 5 is because there are 5 possibilities (1-5)
# 
#     def forward(self, x):
#         # x = F.relu(self.conv1(x))
#         # #x = self.pool(x)
#         # x = F.relu(self.conv2(x))
#         # #x = self.pool(x)
#         # # x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
#         return x


if __name__ == '__main__':
    main()
