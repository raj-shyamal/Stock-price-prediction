import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import yfinance as yf

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

import sys


stocks = ['TSLA']

ohlc = {}


# download stock price data
def download_data():

    for stock in stocks:

        ticker = yf.Ticker(stock)
        ohlc[stock] = ticker.history(period='5y', interval='1d')['Close']

    return pd.DataFrame(ohlc)


# Creating a LSTM (long short term memory) class
class LSTMnet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers, output_size):
        super().__init__()

        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, num_hidden,
                            num_layers, batch_first=True)

        self.out = nn.Linear(num_hidden, output_size)

    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(
            0), self.num_hidden).requires_grad_()
        c = torch.zeros(self.num_layers, x.size(
            0), self.num_hidden).requires_grad_()

        y, (h_, c_) = self.lstm(x, (h.detach(), c.detach()))

        o = self.out(y)
        return o, (h_, c_)


df = download_data()

print(df)


# plotting the stock prices
plt.plot(df)
plt.title('TSLA stock price')
plt.xlabel('time')
plt.ylabel('stock price')
plt.show()


# defining train and test data
propTraining = .8
nTraining = int(len(df)*propTraining)

train_data = df[:nTraining]
test_data = df[nTraining:]


# using MinMaxScaler to scale data between -1 and 1
scalar = MinMaxScaler(feature_range=(-1, 1))

train_data = scalar.fit_transform(train_data.values.reshape(-1, 1))


# Creating a Tensor from a numpy.ndarray.
data = torch.from_numpy(train_data).float()
data = data.reshape([len(data)])


# Network parameters
input_size = 1
hidden_size = 20
num_layers = 2
output_size = 1
seqlength = 7
batchsize = 1


N = len(data)

numepochs = 100


# mean squared error loss function
lossfun = nn.MSELoss()


# creating an instance of the LSTM class
net = LSTMnet(input_size, hidden_size, num_layers, output_size)

# Using Stochastic Gradient Descent to optimize the network parameters
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

losses = np.zeros(numepochs)


# Training the model
for epochI in range(numepochs):

    seglosses = []
    segacc = []
    hidden_states = torch.zeros(num_layers, batchsize, hidden_size)

    for timeI in range(N-seqlength):
        X = data[timeI:timeI+seqlength].view(seqlength, 1, 1)
        y = data[timeI+seqlength].view(1, 1)

        yHat, hidden_states = net(X)
        finalValue = yHat[-1]

        loss = lossfun(finalValue, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        seglosses.append(loss.item())

    losses[epochI] = np.mean(seglosses)

    msg = f'Finished epoch {epochI+1}/{numepochs}'
    sys.stdout.write('\r'+msg)


# plotting the losses
plt.plot(losses, 's-')
plt.title("loss as a function of number of epochs")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# Final forward pass through the network for train data

h = np.zeros((N, seqlength, hidden_size))
c = np.zeros((N, seqlength, hidden_size))

yHat = np.zeros(N)/0

for timeI in range(N-seqlength):
    X = data[timeI:timeI+seqlength].view(seqlength, 1, 1)

    yy, (hn, cn) = net(X)
    yHat[timeI+seqlength] = yy[-1]
    h[timeI+seqlength] = hn[0].detach()


# actual and predicted values
plt.figure(figsize=(25, 4))
plt.plot(data, label='actual data')
plt.plot(yHat, label='predicted')
plt.title('Train Data: actual and predicted')
plt.legend()
plt.show()


plt.plot(data-yHat, 'k^')
plt.title('Train Data: Errors')
plt.xlabel('time')
plt.ylabel('error')
plt.show()


r = np.corrcoef(data[seqlength:], yHat[seqlength:])[0, 1]


plt.plot(data[seqlength:], yHat[seqlength:], 'mo')
plt.title(f'correlation coefficient for train data = {r:.2f}')
plt.xlabel('real data')
plt.ylabel('predicted data')
plt.show()


print(f'correlation coefficient for train data = {r:.2f}')


plt.figure(figsize=(16, 5))
plt.plot(h[:, 0, :])
plt.title('Train Data: hidden states')
plt.xlabel('time')
plt.ylabel('hidden state')
plt.show()


# Testing the network on test data

test_data = scalar.fit_transform(test_data.values.reshape(-1, 1))


data = torch.from_numpy(test_data).float()
data = data.reshape([len(data)])

N = len(data)


h_test = np.zeros((N, seqlength, hidden_size))
c_test = np.zeros((N, seqlength, hidden_size))

yHat = np.zeros(N)/0

for timeI in range(N-seqlength):
    X = data[timeI:timeI+seqlength].view(seqlength, 1, 1)

    yy, (hn, cn) = net(X)
    yHat[timeI+seqlength] = yy[-1]
    h[timeI+seqlength] = hn[0].detach()


plt.figure(figsize=(25, 4))
plt.plot(data, label='actual data')
plt.plot(yHat, label='predicted')
plt.title('Test Data: actual and predicted')
plt.legend()
plt.show()


plt.plot(data-yHat, 'k^')
plt.title('Test Data: Errors')
plt.xlabel('time')
plt.ylabel('error')
plt.show()


r = np.corrcoef(data[seqlength:], yHat[seqlength:])[0, 1]


plt.plot(data[seqlength:], yHat[seqlength:], 'mo')
plt.title(f'correlation coefficient for Test data = {r:.2f}')
plt.xlabel('real data')
plt.ylabel('predicted data')
plt.show()

print(f'correlation coefficient for Test data = {r:.2f}')


plt.figure(figsize=(16, 5))
plt.plot(h[:, 0, :])
plt.title('Test data: hidden states')
plt.xlabel('time')
plt.ylabel('hidden state')
plt.show()
