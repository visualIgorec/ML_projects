from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class CreditScoreModel:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train


    def log_reg(self):
        parameters_lr = {'penalty' :['l2'],
                         'tol' :[1e-4],
                         'solver' :['sag', 'saga'],
                         }
        clf_log_reg = GridSearchCV(LogisticRegression(max_iter=10000, random_state=101), parameters_lr, refit=True)
        clf_log_reg.fit(self.x_train, self.y_train)
        return clf_log_reg.best_estimator_


    def svc(self):
        parameters_svc = {'kernel' :['poly', 'sigmoid'],
                          'degree' :[2, 4, 6],
                          'tol' :[1e-4],
                          }
        clf_svc = GridSearchCV(SVC(max_iter=10000, random_state=101), parameters_svc, refit=True)
        clf_svc.fit(self.x_train, self.y_train)
        return clf_svc.best_estimator_


    def decision_tree(self):
        parameters_dt = {'criterion' :['entropy', 'gini'],
                         'splitter' :['best', 'random']
                         }
        clf_dt = GridSearchCV(DecisionTreeClassifier(random_state=101), parameters_dt, refit=True, error_score='raise')
        clf_dt.fit(self.x_train, self.y_train)
        return clf_dt.best_estimator_