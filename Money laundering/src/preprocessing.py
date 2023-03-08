import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class PrepareData():

    def __init__(self, train, test, valid_size=0.3, random_number=42, norm=True):
        self.valid_size = valid_size
        self.random_number = random_number
        self.norm = norm
        self.train = train
        self.test = test

    def main_process(self):
        # nan to zero
        self.train = self.train.fillna(0)
        self.test = self.test.fillna(0)

        # category encoding
        train_object_list = list(self.train.select_dtypes(include=['object']))
        test_object_list = list(self.test.select_dtypes(include=['object']))

        ord_enc = OrdinalEncoder()
        for itm in train_object_list:
            self.train[itm] = ord_enc.fit_transform(self.train[[itm]])
        for itm in test_object_list:
            self.test[itm] = ord_enc.fit_transform(self.test[[itm]])

        # split to train/valid data
        y_train = self.train['target']
        self.train.drop(['user_id', 'target'], axis=1, inplace=True)
        id_person_test = self.test['Id']
        self.test.drop(['Id', 'user_id'], axis=1, inplace=True)

        x_train, x_val, y_train, y_val = train_test_split(
            self.train, 
            y_train, 
            train_size = self.valid_size, 
            random_state = self.random_number,
            shuffle=True
            )

        # normalization data
        if self.norm:
            scaler_train = preprocessing.StandardScaler().fit(x_train)
            scaler_valid = preprocessing.StandardScaler().fit(x_val)
            scaler_test = preprocessing.StandardScaler().fit(self.test)

            x_train = scaler_train.transform(x_train)
            x_val = scaler_valid.transform(x_val)
            x_test = scaler_test.transform(self.test)

        return {'x_train': x_train, 'x_val': x_val, 'x_test': x_test, 'y_train': y_train, 'y_val': y_val, 'id_person_test': id_person_test}
