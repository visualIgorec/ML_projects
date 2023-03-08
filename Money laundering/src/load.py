import pandas as pd


class LoadData():

    def load(path='data/train.csv'):
        data = pd.read_csv(path, sep=',')
        return data