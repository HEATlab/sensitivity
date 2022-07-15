import glob
import json
from logging import raiseExceptions
import math
import os
from scipy.stats import gamma

def changetimes(data_path):
    data_list = glob.glob(os.path.join(data_path, '*.json'))
    for file in data_list:
        changeSingleTime(file)

def changeToGamma(data_path):
    data_list = glob.glob(os.path.join(data_path, '*.json'))
    for file in data_list:
        print(file)
        changeSingleGamma(file)

def changeSingleTime(testJson):
    with open(testJson, 'r') as f:
        jsonSTN = json.loads(f.read())

        for v in jsonSTN['nodes']:
            if 'min_domain' in v:
                v['min_domain'] *= 1000
            if 'max_domain' in v and v['max_domain'] != 'inf':
                v['max_domain'] *= 1000

        for e in jsonSTN['constraints']:
            if 'min_duration' in e:
                e['min_duration'] *= 1000
            if 'max_duration' in e and e['max_duration'] != 'inf':
                e['max_duration'] *= 1000

    with open(testJson, "w") as jsonFile:
        json.dump(jsonSTN, jsonFile)


def changeSingleGamma(testJson):
    with open(testJson, 'r') as f:
        jsonSTN = json.loads(f.read())
        for e in jsonSTN['constraints']:
            if e['type'] == 'pstc' and 'distribution' in e:
                name_split = e['distribution']['name'].split("_")
                loc = e['min_duration']/1000
                mean = float(name_split[1]) - loc
                sigma = (e['max_duration'] - e['min_duration'])/1000/10
                if sigma == 0:
                    e = {
                        "first_node": e['first_node'],
                        "second_node": e['second_node'],
                        "type": "stc",
                        "min_duration": e['min_duration'],
                        "max_duration": e['max_duration']
                    },
                else:
                    variance = sigma*sigma
                    beta = 1
                    alpha = sigma*sigma 
                    leftBound = gamma.ppf(q= 0.0000000001, a=alpha, scale = 1)
                    loc += leftBound
                    e['distribution'] = {
                        "name": "G_"+str(alpha)+"_("+str(1/beta)+","+str(loc) +")",
                        "type": "Empirical"
                    }

    with open(testJson, "w") as jsonFile:
        json.dump(jsonSTN, jsonFile)