import pickle
from os.path import exists
from datetime import datetime
import pandas as pd
import numpy as np

def write_last_generation(weights):
    with open("models/last_generation.pkl", "wb") as f:
        pickle.dump(weights, f)
        
def read_last_generation():
    with open("models/last_generation.pkl", "rb") as f:
        return pickle.load(f)

def write_information(fitness, step):
    
    if exists("models/information.pkl"):
        information = pd.read_pickle("models/information.pkl")
    else:
        information = pd.DataFrame({
            "datetime": [],
            "fitness": [],
            "step": [],
        })
    
    pd.concat([
        information,
        pd.DataFrame({
            "datetime": [pd.to_datetime(datetime.now())],
            "fitness": [np.average(fitness)],
            "step": [np.average(step)],
        }),
    ]).reset_index(drop=True).to_pickle("models/information.pkl")
    
def read_information():
    return pd.read_pickle("models/information.pkl")