# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from pandas_datareader import data as pdr
import yfinance as yf
import datetime

# create 1 data into time series dataset
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

# for productivity
np.random.seed(7) 

# important dataset
start_=datetime.datetime(2021, 1, 1) 
end_=datetime.datetime(2021, 4, 18)   # Update the time interval you want to use.
yf.pdr_override() 
dataset=pdr.get_data_yahoo('ETH', start_,end_)  # Enter the symbol of the stock you want to predict
dataset=dataset.iloc[:,[0,1,2,3]]


obs = np.arange(1, len(dataset) + 1, 1)

# predict
OHLC_avg = dataset.mean(axis = 1) #average of Open, High, Low and Closing Prices
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1) #average of High, Low and Closing Prices
close_val = dataset[['Close']]

# plot
plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
plt.plot(obs, close_val, 'g', label = 'Closing price')
plt.legend(loc = 'upper right')
plt.show()

# prepare for time series
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) 
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg) #All values have been normalized between 0 and 1.

# train set split
# 75% data is used for training
train_OHLC = int(len(OHLC_avg) * 0.75)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# time series split
# After converting the dataset into OHLC average, it becomes
#  one column data. This has been converted into two column 
# time series data, 1st column consisting stock price of time t, 
# and second column of time t+1.
trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY = new_dataset(test_OHLC, 1)

# reshape train and test data set
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# Two sequential LSTM layers have been stacked together and one dense layer 
# is used to build the RNN model using Keras deep learning library. 
model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
# This is a regression task, 'linear' activation has been used in final layer.
model.add(Activation('linear'))

# modle complie and train
model.compile(loss='mean_squared_error', optimizer='adam') 
# Try SGD, adam, adagrad and compare.Find that adam performs best.
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

# predict
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# de-normalization for plot
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# Test accuracy metric is root mean square error (RMSE)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))

# test RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))


trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict


testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict


OHLC_avg = scaler.inverse_transform(OHLC_avg)


plt.plot(OHLC_avg, 'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'training set')
plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
plt.legend(loc = 'upper right')
plt.xlabel('Time in Days')
plt.ylabel('OHLC Value of  ETH')
plt.show()


last_val = OHLC_avg[-1]
last_val_scaled = scaler.transform([last_val])
next_val_scaled = model.predict(np.reshape(last_val_scaled, (1,1,1)))
next_val=scaler.inverse_transform(next_val_scaled)
print ("Last Day Value:", last_val[0])
print ("Next Day Value:", next_val[0][0])
