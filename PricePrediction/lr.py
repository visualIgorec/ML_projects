import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class AnalysisDataAndFitLinearRegression:

    def __init__(self):
        self.model = None

    def getAndAnaliseData(self, path):
        df = pd.read_csv(path, sep=',')
        name_list = list(df)

        TARGET = name_list[0]
        FEATURES = name_list[1:]

        #sorted_tab
        df_sort = df[
            (df['Bedroom'] >= 2) &
            (df['Price'] >= 50) &
            (df['Condition'] == 1)
        ]
        print(f'Sorted table: {df_sort}')

        # dict with statistical info
        stat_dict = {
            'mean': dict(df.mean(axis=0)),
            'max': dict(df.max(axis=0)),
            'min': dict(df.min(axis=0)),
            'std': dict(df.std(axis=0))
        }

        # drop Nan values
        df = self.dropMissingValues(df)

        # parameters of linear reg
        param_dict = {
            'fit_intercept': True,
            'normalize': 'deprecated',
            'copy_X': True,
            'n_jobs': None,
            'positive': False
        }

        # lr
        X = np.array(df[FEATURES])
        Y = np.array(df[TARGET])

        self.model = LinearRegression(**param_dict).fit(X, Y)
        score = self.model.score(X, Y)

        return {
            'stat_dict': stat_dict,
            'param_dict': param_dict,
            'score': score,
        }

    def predictLinearRegression(self, input_data):
        return self.model.predict(input_data)

    def dropMissingValues(self, data):
        return data.dropna(axis=0)


# Path
path = "./data/realest.csv"

# Create Object for analysis and training
lr_obj = AnalysisDataAndFitLinearRegression()
print(f'Main info: {lr_obj.getAndAnaliseData(path)}')

# Make prediction
x_input = np.array([[3, 1500, 8, 40, 1000, 2, 1, 1]])
print(f'Prediction: {round(lr_obj.predictLinearRegression(x_input)[0], 3)}')
