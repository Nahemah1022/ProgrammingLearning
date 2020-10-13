import sys
import json
import pandas as pd
import os
from os import path
from datetime import datetime, timedelta
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

"""
load data
"""

# p = '/home/db/mlb/res/stock/twse/json/'
p = '/home/nahemah1022/Desktop/Program/ProgrammingLearning/ML&AI/data/stock/json/'

today = '2020-10-08'
dates = []
data = []
checkID = []
done = []
for root, dirs, files in os.walk("./models"):
    for file in files:
        done.append(file.split('_')[0])
if(path.exists(p + today + '.json')):
    with open(p + today + '.json') as f:
        f = json.load(f)
        for key, value in f.items():
            if(key not in done):
                checkID.append(key)
            else:
                print('skip', key)

for i in range(len(checkID)):
    data.append([])

date = datetime.strptime(today, '%Y-%m-%d')

counter = 0
miss = 0
while counter < 10000 and miss < 10000:
    date -= timedelta(1)
    d = date.strftime('%Y-%m-%d')

    if(path.exists(p + d + '.json')):
        print(d)
        dates.append(d)
        counter += 1
        with open(p + d + '.json') as f:
            f = json.load(f)
            for idx, ID in enumerate(checkID):
                if(ID in f and ID != 'id'):
                    for k, v in f[ID].items():
                        if(ID == 'id' or ID not in f or v == 'NULL' or v is None or v == ""):
                            f[ID][k] = np.nan
                        else:
                            f[ID][k] = float(f[ID][k])
                else:
                    f[ID] = {'adj_close': np.nan, 'close': np.nan, 'high': np.nan, 'low': np.nan, 'open': np.nan, 'volume': np.nan}
                data[idx].append(f[ID])
    else:
        miss += 1

for idx, ID in enumerate(checkID):
    print(ID)
    training_data_len = math.ceil( len(dates) *.8)

    df = pd.DataFrame(data[idx], index = dates)
    df = df.interpolate(method ='linear', limit_direction ='forward')
    df = df.interpolate(method ='linear', limit_direction ='backward')
    dataset = df.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0 : training_data_len, : ]
    x_train = []
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i - 60 : i, :])
        y_train.append(train_data[i, :])
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=6))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)


    test_data = scaled_data[training_data_len - 60: , : ]
    x_test = []
    y_test =  dataset[training_data_len : , :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, : ])
    x_test = np.array(x_test)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
    print("rmse:")
    print(rmse)

    #export model
    model.save('models/' + ID + '_close_prediction.h5')