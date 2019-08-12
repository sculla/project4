import pandas as pd


def load_set():
    loaded_df = pd.read_csv('data/data.csv', header=0)
    assert pd.core.frame.DataFrame == type(loaded_df), 'load_set failed to output a df'
    assert loaded_df.__len__() == 36941, 'data is missing, length should be 36941'
    return loaded_df
