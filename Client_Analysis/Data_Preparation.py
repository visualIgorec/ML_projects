from sklearn.preprocessing import StandardScaler
import pandas as pd


class Data_Preparation:

    def __init__(self, x_data, y_data, f_name):
        self.x_data = x_data
        self.y_data = y_data
        self.f_name = f_name

    def run_up(self):
        x_train = self.x_data[self.f_name].iloc[:int(len(self.x_data) * 0.7), :]
        y_train = self.y_data.iloc[:int(len(self.x_data) * 0.7)]
        y_train = y_train.ravel()

        x_test = self.x_data[self.f_name].iloc[int(len(self.x_data) * 0.7):, :]
        y_test = self.y_data.iloc[int(len(self.x_data) * 0.7):]
        y_test = y_test.ravel()

        assert len(x_train) == len(y_train), print('Error in Shape')
        assert len(x_test) == len(y_test), print('Error in Shape')

        # Нормализация входных признаков
        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)

        return x_train, x_test, y_train, y_test