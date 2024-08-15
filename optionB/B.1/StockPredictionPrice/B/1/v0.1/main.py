# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM # long short term memory layers

# Step 2: Load Data
company = '2330.TW' # specify a company

start = dt.datetime(2010, 1, 1) # time point want to start
end = dt.datetime(2020, 1, 1) # should not use until now

data = yf.download(company, start=start, end=end) # load the ticket symbol company


# Step 3: Prepare data
scaler = MinMaxScaler(feature_range=(0, 1)) # Scale down to fit 0 and 1
scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1)) #Predict the market have closed

predictionDays = 60 # how many days to look back

xTrain = []
yTrain = []

# start counting 60 from the index
for x in range(predictionDays, len(scaledData)):
    xTrain.append(scaledData[x-predictionDays:x, 0]) # labeled data, also to preapre 61st of the data
    yTrain.append(scaledData[x, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1)) # add 1 additional dimensional

# Step 4: Build the models
model = Sequential()

# Always add one LSTM layer ->? dropout layer -> so on and so forth until Dense layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(xTrain.shape[1], 1))) #LSTM is a recurrent cell so it's going to feedback the info, unlike feed forward info like dense layer
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True)) # Remove input shape cause we don't need that
model.add(Dropout(0.2))
model.add(LSTM(units=50)) # Not going to return sequence
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of the next closing value


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain, epochs=25, batch_size=32) #model going to see 24 times, see 32 units at once

'''Test the Model Accuracy on Existing Data'''

# Step 5: Load Test Data
testStart = dt.datetime(2020, 1, 1)
testEnd = dt.datetime.now() # Time range of the test data

testData = yf.download(company, start=testStart, end=testEnd) # data reader company, meta finance api
actualPrices = testData['Close'].values

# Combine training and test data, concatenate close value of data, close value of the test data
totalDataset = pd.concat((data['Close'], testData['Close']), axis = 0)

# Predict the next price, evaluate how accurate the model
modelInputs = totalDataset[len(totalDataset) - len(testData) - predictionDays:].values # colon (:) up until the end
modelInputs = modelInputs.reshape(-1, 1)
modelInputs = scaler.transform(modelInputs)

# Step 6: Make Prediction on Test data

xTest = []

for x in range(predictionDays, len(modelInputs)):
    xTest.append(modelInputs[x-predictionDays:x, 0]) # Not get numbers until X

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

predictedPrices = model.predict(xTest)
predictedPrices = scaler.inverse_transform(predictedPrices) # Using scaler, back to actual predicted prices here

# Plot the Test Prediction
plt.plot(actualPrices, color='black', label=f'Actual {company} Price') #
plt.plot(predictedPrices, color='green', label=f'Predicted  {company} Price') # Not going to be the same because accuracy is not 100 accurate
plt.title(f"{company} Share price")
plt.xlabel("Time")
plt.ylabel(f"{company} Share price")
plt.legend()
plt.show()

# Step 7: Predict the following day

realData = [modelInputs[len(modelInputs) + 1 - predictionDays:len(modelInputs+1), 0]]
realData = np.array(realData)
realData = np.reshape(realData, (realData.shape[0], realData.shape[1], 1))

prediction = model.predict(realData) # use real data as the input
prediction = scaler.inverse_transform(prediction)

print(f"Prediction: {prediction}")




