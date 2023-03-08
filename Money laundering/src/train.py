from sklearn.linear_model import LogisticRegression


class ClassificationModel():

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        # logreg
        clf_logreg = LogisticRegression().fit(self.x_train, self.y_train)
        return clf_logreg