# Stock Price Prediction Using LSTM

Applied a multi-layer long short-term memory (LSTM) RNN to the input 'TSLA' stock close price for a period of 5 years.

![](./plots/TSLA%20stock%20price.png)

For each element in the input sequence, each layer computes the following function:

![](./lstm%20architecture.jpg)

$$
i_t = \sigma(W_{ii}x_t + b_{ii} + W_{hi}h_{(t-1)} + b_{hi})
$$

$$
f_t = \sigma(W_{if}x_t + b_{if} + W_{hf}h_{(t-1)} + b_{hf})
$$

$$
g_t = \tanh(W_{ig}x_t + b_{ig} + W_{hg}h_{(t-1)} + b_{hg})
$$

$$
o_t = \sigma(W_{io}x_t + b_{io} + W_{ho}h_{(t-1)} + b_{ho})
$$

$$
c_t = f_t * c_{(t-1)} + i_t * g_t
$$

$$
h_t = o_t * \tanh(c_t)
$$

## Results

After training the network:

- The correlation coefficient between real values and predicted values for train data is 1.00

![](./plots/train%20data%20corr%20coef.png)

- The correlation coefficient between real values and predicted values for test data is 0.97

![](./plots/test%20data%20corr%20coef.png)
