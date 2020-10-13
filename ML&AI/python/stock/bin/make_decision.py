import sys
import os
import json
import random
import numpy as np
from predict import predictById

checkID = []
if(os.path.exists('./json/' + '2020-10-08' + '.json')):
    with open('./json/' + '2020-10-08' + '.json') as f:
        f = json.load(f)
        for key, value in f.items():
            if(key != 'id'):
                checkID.append(key)
output = []

for id in checkID[:15]:
    rtn = predictById(id, sys.argv[1], 5)
    if(not rtn[0]):
        continue
    base = rtn[0]
    pred = rtn[1]
    # print(id)
    # print(base)
    # print(pred)
    max_diff = 0
    value = 0
    count = 0
    max_idx = 1
    break_flag = False

    for date, close in pred.items():
        count += 1
        # print(close)
        if(np.isnan(close)):
            break_flag = True
            break
        if(max_diff < abs(close - base[1])):
            max_diff = abs(close - base[1])
            value = close
            max_idx = count
    if(break_flag):
        continue
    obj = {
        "code": id,
        "type": "buy" if (value - base[1]) > 0 else "short",
        "open_price": base[1] + (value - base[1]) / 4,
        "weigth": 1,
        "close_high_price": base[1] + (value - base[1]) * 3 / 4,
        "close_low_price": base[1] - (value - base[1]) * 1 / 2,
        "life": max_idx
    }
    output.append(obj)

f = open(os.getcwd() + '/../commit/' + sys.argv[1] + '_' + sys.argv[1] + '.json', 'w')
json.dump(output, f, indent=4)
f.close()