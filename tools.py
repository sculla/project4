import pandas as pd
import numpy as np


def load_set():
    return pd.read_csv('data/data.csv', header=0)
