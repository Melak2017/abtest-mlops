import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import sys
sys.setrecursionlimit(5000)

def loss_function(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse

def root_mean_squared_error(act, pred):
 
   rmse = np.sqrt(mean_squared_error(act, pred))
   return rmse

def mean_absolute_error(act, pred):
    mae = mean_absolute_error(act, pred)
    return mae

def get_metrics(act, pred):
        acc = accuracy_score(act, pred)
        prec = precision_score(act, pred)
        recall = recall_score(act, pred)
        confusion = True
        try:
            cm = confusion_matrix(act, pred)
            true_pos = cm[0][0]
            true_neg = cm[1][1]
            false_pos = cm[0][1]
            false_neg = cm[1][0]
        except:
            print("Unable to calculate confusion matrix")
            confusion = False
        metric = {
            'accuracy': round(acc, 2),
            'precision': round(prec, 2),
            'recall': round(recall, 2)
        }
        if confusion:
            metric['true_neg'] = true_neg
            metric['true_pos'] = true_pos
            metric['false_neg'] = false_neg
            metric['false_pos'] = false_pos
        
        return metric


class DecisionTreeModel:
    
    def __init__(self, X_train, X_test, y_train, y_test, max_depth=5):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.loss_function = None
        
        self.model = DecisionTreeClassifier(max_depth=4)

        
    def train(self, folds=1,experiment_name=""):
        
        kf = KFold(n_splits = folds)
        
        iterator = kf.split(self.X_train)
        
        loss_arr = []
        acc_arr = []
        loss_dict = dict()
        loss_dict['mae'] = []
        loss_dict['rmse'] = []
        loss_dict['mse'] = []
        mlflow.set_experiment(experiment_name)
        for i in range(folds):
            train_index, valid_index = next(iterator)
            
            X_train, y_train = self.X_train.iloc[train_index], self.y_train.iloc[train_index]
            X_valid, y_valid = self.X_train.iloc[valid_index], self.y_train.iloc[valid_index]

            with mlflow.start_run(run_name=f"{i} fold training"):
                mlflow.sklearn.autolog()
                self.model = self.model.fit(X_train, y_train)
                
                vali_pred = self.model.predict(X_valid)
                
                accuracy = self.calculate_score(y_valid
                                                , vali_pred)
                
                loss = loss_function(y_valid, vali_pred)
                # loss_dict['mae'].append(mean_absolute_error(y_valid, vali_pred))
                # loss_dict['rmse'].append(root_mean_squared_error(y_valid, vali_pred))
                # loss_dict['mse'].append(mean_squared_error(y_valid, vali_pred))
                # mlflow.log_metric("rmse", loss)
                # mlflow.log_metric("r2", r2_score(y_valid, vali_pred))
                # mlflow.log_metric("mae", mean_absolute_error(y_valid, vali_pred))
                
                self.__printAccuracy(accuracy, i, label="Validation")
                self.__printLoss(loss, i, label="Validation")
                print()
                
                acc_arr.append(accuracy)
                loss_arr.append(loss)
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                # Model registry does not work with file store
                # if tracking_url_type_store != "file":

                #     # Register the model
                #     # There are other ways to use the Model Registry, which depends on the use case,
                #     # please refer to the doc for more information:
                #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                #     mlflow.sklearn.log_model(self.model, "model", registered_model_name="DecisionTreeModel")
                # else:
                #     mlflow.sklearn.log_model(self.model, "model")

        #  self.log_model()   
        return self.model, acc_arr, loss_arr
    
    def test(self):
        
        y_pred = self.model.predict(self.X_test)
        
        accuracy = self.calculate_score(y_pred, self.y_test)
        self.__printAccuracy(accuracy, label="Test")
        
        report = self.report(y_pred, self.y_test)
        matrix = self.confusion_matrix(y_pred, self.y_test)
        
        loss = loss_function(self.y_test, y_pred)
        
        return accuracy, loss,  report, matrix
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        fi_df = pd.DataFrame()
        
        fi_df['feature'] = self.X_train.columns.to_list()
        fi_df['feature_importances'] = importance
        figure = plt.figure(figsize=(10,10))
        ax = sns.barplot(x=fi_df['feature'], y=fi_df['feature_importances'])
        ax.set_title('Feature Importance')
        ax.set_xlabel("Features", fontsize=18)
        ax.set_ylabel("Feature Importance", fontsize=18)
        
        return fi_df, figure
    
    def __printAccuracy(self, acc, step=1, label=""):
        print(f"step {step}: {label} Accuracy of DecisionTreesModel is: {acc:.3f}")
    
    def __printLoss(self, loss, step=1, label=""):
        print(f"step {step}: {label} Loss of DecisionTreesModel is: {loss:.3f}")
    
    def calculate_score(self, pred, actual):
        return metrics.accuracy_score(actual, pred)
    
    def report(self, pred, actual):
        print("Test Metrics")
        print("================")
        print(metrics.classification_report(pred, actual))
        return metrics.classification_report(pred, actual)
    
    def confusion_matrix(self, pred, actual):
        figure = plt.figure(figsize=(10, 10))
        ax=sns.heatmap(pd.DataFrame(metrics.confusion_matrix(pred, actual)))
        plt.title('Confusion matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        return metrics.confusion_matrix(pred, actual),figure

    def log_model(self, x_test,y_test, experiment_name, run_name, model_name, run_params=None):
        pred = self.model.predict(x_test)
        run_metrics = get_metrics(y_test, pred)
        feature_importance, feature_importance_plot = self.get_feature_importance()
        _, cm_plot = self.confusion_matrix(pred, y_test)


        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri('http://localhost:5000')
        with mlflow.start_run(run_name=run_name):
            if run_params:
                for name in run_params:
                    mlflow.log_param(name, run_params[name])
            for name in run_metrics:
                mlflow.log_metric(name, run_metrics[name])

            mlflow.log_param("columns", x_test.columns.to_list())
            mlflow.log_figure(cm_plot, "confusion_matrix.png")
            mlflow.log_figure(feature_importance_plot,
                              "feature_importance.png")
            cm_plot.savefig("../images/confusion_matrix.png")
            feature_importance_plot.savefig("../images/feature_importance.png")
            mlflow.log_dict(feature_importance.to_json(), "feature_importance.json")

            mlflow.sklearn.log_model(
                sk_model=self.model, artifact_path='models', registered_model_name=model_name)
            print('Run - %s is logged to Experiment - %s' %
                (run_name, experiment_name))
        return run_metrics



class XGBClassifierModel:
    
    def __init__(self, X_train, X_test, y_train, y_test, max_depth=5):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model = GradientBoostingClassifier()


        
    def train(self, folds=1, experiment_name=""):
        
        kf = KFold(n_splits = folds)
        
        iterator = kf.split(self.X_train)
        
        loss_arr = []
        acc_arr = []
        mlflow.set_experiment(experiment_name)
        for i in range(folds):
            train_index, valid_index = next(iterator)
            
            X_train, y_train = self.X_train.iloc[train_index], self.y_train.iloc[train_index]
            X_valid, y_valid = self.X_train.iloc[valid_index], self.y_train.iloc[valid_index]

            with mlflow.start_run(run_name=f"{i} fold training"):
                mlflow.sklearn.autolog()
                        
                self.model = self.model.fit(X_train, y_train)
                
                vali_pred = self.model.predict(X_valid)
                
                accuracy = self.calculate_score(y_valid
                                                , vali_pred)
                
                loss = loss_function(y_valid, vali_pred)
                
                self.__printAccuracy(accuracy, i, label="Validation")
                self.__printLoss(loss, i, label="Validation")
                print()
                
                acc_arr.append(accuracy)
                loss_arr.append(loss)

            
        return self.model, acc_arr, loss_arr
    
    def test(self):
        
        y_pred = self.model.predict(self.X_test)
        
        accuracy = self.calculate_score(y_pred, self.y_test)
        self.__printAccuracy(accuracy, label="Test")
        
        report = self.report(y_pred, self.y_test)
        matrix = self.confusion_matrix(y_pred, self.y_test)
        
        loss = loss_function(self.y_test, y_pred)
        
        return accuracy, loss,  report, matrix
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        fi_df = pd.DataFrame()
        
        fi_df['feature'] = self.X_train.columns.to_list()
        fi_df['feature_importances'] = importance
        figure = plt.figure(figsize=(10,10))
        ax = sns.barplot(x=fi_df['feature'], y=fi_df['feature_importances'])
        ax.set_title('Feature Importance')
        ax.set_xlabel("Features", fontsize=18)
        ax.set_ylabel("Feature Importance", fontsize=18)
        
        return fi_df, figure
    
    def __printAccuracy(self, acc, step=1, label=""):
        print(f"step {step}: {label} Accuracy of GradientBoostingClassifier is: {acc:.3f}")
    
    def __printLoss(self, loss, step=1, label=""):
        print(f"step {step}: {label} Loss of GradientBoostingClassifier is: {loss:.3f}")
    
    def calculate_score(self, pred, actual):
        return metrics.accuracy_score(actual, pred)
    
    def report(self, pred, actual):
        print("Test Metrics")
        print("================")
        print(metrics.classification_report(pred, actual))
        return metrics.classification_report(pred, actual)
    
    def confusion_matrix(self, pred, actual):
        figure = plt.figure(figsize=(10, 10))
        ax=sns.heatmap(pd.DataFrame(metrics.confusion_matrix(pred, actual)))
        plt.title('Confusion matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        return metrics.confusion_matrix(pred, actual),figure

    def log_model(self, x_test,y_test, experiment_name, run_name, model_name, run_params=None):
        pred = self.model.predict(x_test)
        run_metrics = get_metrics(y_test, pred)
        feature_importance, feature_importance_plot = self.get_feature_importance()
        _, cm_plot = self.confusion_matrix(pred, y_test)


        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri('http://localhost:5000')
        with mlflow.start_run(run_name=run_name):
            if run_params:
                for name in run_params:
                    mlflow.log_param(name, run_params[name])
            for name in run_metrics:
                mlflow.log_metric(name, run_metrics[name])

            mlflow.log_param("columns", x_test.columns.to_list())
            mlflow.log_figure(cm_plot, "confusion_matrix.png")
            mlflow.log_figure(feature_importance_plot,
                              "feature_importance.png")
            cm_plot.savefig("../images/confusion_matrix.png")
            feature_importance_plot.savefig("../images/feature_importance.png")
            mlflow.log_dict(feature_importance.to_json(), "feature_importance.json")

            mlflow.sklearn.log_model(
                sk_model=self.model, artifact_path='models', registered_model_name=model_name)
            print('Run - %s is logged to Experiment - %s' %
                (run_name, experiment_name))
        return run_metrics


class LogisticRegressionModel:
    
    def __init__(self, X_train, X_test, y_train, y_test, model_name="LR"):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model_name
        
        self.model = LogisticRegression()

        
    def train(self, folds=1, experiment_name=""):
        
        kf = KFold(n_splits = folds)
        
        iterator = kf.split(self.X_train)
        
        loss_arr = []
        acc_arr = []

        mlflow.set_experiment(experiment_name)
        for i in range(folds):

            train_index, valid_index = next(iterator)

            X_train, y_train = self.X_train.iloc[train_index], self.y_train.iloc[train_index]
            X_valid, y_valid = self.X_train.iloc[valid_index], self.y_train.iloc[valid_index]

            with mlflow.start_run(run_name=f"{i} fold training"):
                mlflow.sklearn.autolog()

                self.model = self.model.fit(X_train, y_train)

                vali_pred = self.model.predict(X_valid)

                accuracy = self.calculate_score(y_valid, vali_pred)
                loss = loss_function(y_valid, vali_pred)

                self.__printAccuracy(accuracy, i, label="Validation")
                self.__printLoss(loss, i, label="Validation")
                print()

                acc_arr.append(accuracy)
                loss_arr.append(loss)
            
        return self.model, acc_arr, loss_arr
    
    def test(self):
        y_pred = self.model.predict(self.X_test)
        
        accuracy = self.calculate_score(self.y_test, y_pred)
        self.__printAccuracy(accuracy, label="Test")
        
        report = self.report(y_pred, self.y_test)
        matrix = self.confusion_matrix(y_pred, self.y_test)
        loss = loss_function(self.y_test, y_pred)
        
        return accuracy, loss, report, matrix 
    
    def __printAccuracy(self, acc, step=1, label=""):
        print(f"step {step}: {label} Accuracy of LogesticRegression is: {acc:.3f}")
    
    def __printLoss(self, loss, step=1, label=""):
        print(f"step {step}: {label} Loss of LogesticRegression is: {loss:.3f}")
    
    def calculate_score(self, pred, actual):
        return metrics.accuracy_score(actual, pred)

    def get_feature_importance(self):
        importance = self.model.coef_[0]
        fi_df = pd.DataFrame()
        
        fi_df['feature'] = self.X_train.columns.to_list()
        fi_df['feature_importances'] = importance
        figure = plt.figure(figsize=(10,10))
        ax = sns.barplot(x=fi_df['feature'], y=fi_df['feature_importances'])
        ax.set_title('Feature Importance')
        ax.set_xlabel("Features", fontsize=18)
        ax.set_ylabel("Feature Importance", fontsize=18)
        
        return fi_df, figure
    
    def report(self, pred, actual):
        print("Test Metrics")
        print("================")
        print(metrics.classification_report(pred, actual))
        return metrics.classification_report(pred, actual)
    
    def confusion_matrix(self, pred, actual):
        figure = plt.figure(figsize=(10, 10))
        ax=sns.heatmap(pd.DataFrame(metrics.confusion_matrix(pred, actual)))
        plt.title('Confusion matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        return metrics.confusion_matrix(pred, actual),figure
    
    def get_p_values(self):
        """ 
        Calcualting p_values for logestic regression.
        code refered from the following link
        https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d
        
        """
        denom = (2.0*(1.0+np.cosh(self.model.decision_function(self.X_train))))
        denom = np.tile(denom,(self.X_train.shape[1],1)).T
        F_ij = np.dot((self.X_train/denom).T,self.X_train) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0]/sigma_estimates # z-score 
        p_values = [stat.norm.sf(abs(x)) for x in z_scores] ### two tailed test for p-values
        
        p_df = pd.DataFrame()
        p_df['features'] = self.X_train.columns.to_list()
        p_df['p_values'] = p_values
        
        return p_df
    
    def plot_pvalues(self, p_df):
        
        fig, ax = plt.subplots(figsize=(12,7))

        ax.plot([0.05,0.05], [0.05,5])
        sns.scatterplot(data=p_df, y='features', x='p_values', color="green")
        plt.title("P values of features", size=20)

        plt.xticks(np.arange(0,max(p_df['p_values']) + 0.05, 0.05))

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.show()
        return fig

    def log_model(self, x_test,y_test, experiment_name, run_name, model_name, run_params=None):
        pred = self.model.predict(x_test)
        run_metrics = get_metrics(y_test, pred)
        feature_importance, feature_importance_plot = self.get_feature_importance()
        _, cm_plot = self.confusion_matrix(pred, y_test)


        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri('http://localhost:5000')
        with mlflow.start_run(run_name=run_name):
            if run_params:
                for name in run_params:
                    mlflow.log_param(name, run_params[name])
            for name in run_metrics:
                mlflow.log_metric(name, run_metrics[name])

            mlflow.log_param("columns", x_test.columns.to_list())
            mlflow.log_figure(cm_plot, "confusion_matrix.png")
            mlflow.log_figure(feature_importance_plot,
                              "feature_importance.png")
            cm_plot.savefig("../images/confusion_matrix.png")
            feature_importance_plot.savefig("../images/feature_importance.png")
            mlflow.log_dict(feature_importance.to_json(), "feature_importance.json")

            mlflow.sklearn.log_model(
                sk_model=self.model, artifact_path='models', registered_model_name=model_name)
            print('Run - %s is logged to Experiment - %s' %
                (run_name, experiment_name))
        return run_metrics