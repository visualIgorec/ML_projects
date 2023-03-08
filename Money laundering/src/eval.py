from sklearn.metrics import roc_auc_score
import pandas as pd


class Eval():

    def _eval(clf_logreg, x_val, y_val, x_test, id_person_test):
        
        # check on valid set
        valid_score = roc_auc_score(y_val, clf_logreg.predict_proba(x_val)[:,1])
        print(f'roc_auc_score on valid set: {valid_score}')

        # predict and save in .csv
        test_prediction = clf_logreg.predict(x_test)
        new_df = pd.DataFrame(test_prediction)
        new_df.to_csv('pred.csv')