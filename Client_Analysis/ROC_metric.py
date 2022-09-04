from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


class ROC_metric:

    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test


    def roc_lg(self, lg_reg_classificator, y_pred_lg_reg):
        plot_confusion_matrix(lg_reg_classificator, self.x_test, self.y_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_lg_reg)
        score_auc = roc_auc_score(self.y_test, y_pred_lg_reg)
        plt.title('Confusion Matrix for LogisticRegression')
        plt.show()
        plt.plot(fpr, tpr, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for LogisticRegression. AUC area = {round(score_auc ,3)}')
        plt.legend()
        plt.show()


    def roc_svc(self, svc_classificator, y_pred_lg_reg):
        plot_confusion_matrix(svc_classificator, self.x_test, self.y_test)
        plt.title('Confusion Matrix for SVC')
        plt.show()
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_lg_reg)
        score_auc = roc_auc_score(self.y_test, y_pred_lg_reg)
        plt.plot(fpr, tpr, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for SVC. AUC area = {round(score_auc ,3)}')
        plt.legend()
        plt.show()


    def roc_dt(self, dt_classificator, y_pred_lg_reg):
        plot_confusion_matrix(dt_classificator, self.x_test, self.y_test)
        plt.title('Confusion Matrix for Decision Tree')
        plt.show()
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_lg_reg)
        score_auc = roc_auc_score(self.y_test, y_pred_lg_reg)
        plt.plot(fpr, tpr, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Decision Tree. AUC area = {round(score_auc ,3)}')
        plt.legend()
        plt.show()