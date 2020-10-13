from load import load_test, get_base_date
from os import path
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

def predictById(stockID, today, pred_days):
    date = datetime.strptime(today, '%Y-%m-%d')

    predictions = []
    output = {}
    pair = get_base_date(today, stockID)

    if(path.exists('./models/' + stockID + '_close_prediction.h5')):
        model = keras.models.load_model('./models/' + stockID + '_close_prediction.h5')
        test_data = load_test(today, stockID)

        for progress in range(pred_days):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(test_data.values)

            x_test = [scaled_data]
            x_test = np.array(x_test)

            prediction = model.predict(x_test)
            prediction = scaler.inverse_transform(prediction)
            predictions.append(prediction[0])

            # shift and insert new row
            test_data = test_data.iloc[:-1]
            obj = {
                "adj_close": prediction[0][0], 
                "close": prediction[0][1], 
                "high": prediction[0][2], 
                "low": prediction[0][3], 
                "open": prediction[0][4], 
                "volume": prediction[0][5]
            }
            row = pd.DataFrame(obj, index = [today]) 
            test_data = pd.concat([row, test_data])

            output[today] = prediction[0][1]

            date += timedelta(1)
            today = date.strftime('%Y-%m-%d')
        return (pair, output)
    else: 
        return (False, False)