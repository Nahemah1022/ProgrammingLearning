import json
import sys
import numpy as np
import pandas as pd
from os import path
from datetime import datetime, timedelta

p = './json/'

def get_base_date(start_date, stockID):
    date = datetime.strptime(start_date, '%Y-%m-%d')
    while True:
        date -= timedelta(1)
        d = date.strftime('%Y-%m-%d')
        if(path.exists(p + d + '.json')):
            with open(p + d + '.json') as f:
                f = json.load(f)
                if(stockID != 'id' and stockID in f and f[stockID]['close'] != 'NULL' and f[stockID]['close'] is not None and f[stockID]['close'] != ""):
                    return (d, float(f[stockID]['close']))


def load_test(start_date, stockID):
    date = datetime.strptime(start_date, '%Y-%m-%d')
    date -= timedelta(1)

    counter = 0
    data = []
    dates = []

    while counter < 60:
        d = date.strftime('%Y-%m-%d')
        if(path.exists(p + d + '.json')):
            dates.append(d)
            with open(p + d + '.json') as f:
                f = json.load(f)
                for k, v in f[stockID].items():
                    if(stockID == 'id' or stockID not in f or v == 'NULL' or v is None or v == ""):
                        f[stockID][k] = np.nan
                    else:
                        f[stockID][k] = float(f[stockID][k])
                data.append(f[stockID])
                counter += 1
                    
        date -= timedelta(1)
    return pd.DataFrame(data, index = dates)
