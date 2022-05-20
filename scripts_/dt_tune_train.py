from app_logger import App_Logger
from helper import Helper
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import dvc.api
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split


import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join('../scripts')))


helper = Helper()

logger = App_Logger("models.log").get_app_logger()


params = {'criterion': ['gini', 'entropy'],
          'max_depth': [4, 5, 6, 7, 8, 9, 10]}


def get_data(tag, path='Data/clean_data.csv', repo='https://github.com/daniEL2371/abtest-mlops'):
    rev = tag
    data_url = dvc.api.get_url(path=path, repo=repo, rev=rev)
    df = pd.read_csv(data_url)
    return df


def split_train_test(X, y, test_size=0.1):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def loss_function(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return rmse


def DecisionTree_tune(clf, splitted_data, loss_function, folds=5, params=params, run_name='DT-Hyperparameter', save=False, save_path=None):

    kfold = KFold(n_splits=folds)
    gridSearch = GridSearchCV(estimator=clf, param_grid=params,
                              n_jobs=-1,  cv=kfold, scoring="neg_root_mean_squared_error")

    X_train, X_test, y_train, y_test = splitted_data

    mlflow.set_experiment('DecisionTree-tune')
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        searchResults = gridSearch.fit(X_train, y_train)

        pred = searchResults.predict(X_test)
        loss = loss_function(y_test, pred)
        acc = metrics.accuracy_score(y_test, pred)

        mlflow.log_param('Features', X_train.columns.to_list())
        mlflow.log_param('Target', y_train.columns.to_list())
        mlflow.log_param('Number Of Training Dataset', X_train.shape[0])
        mlflow.log_param('Number Of Test Dataset', X_test.shape[0])
        mlflow.log_param('Fold number', folds)

        mlflow.log_metric("loss", loss)
        mlflow.log_metric("accuracy", acc)

    best_dt_Model = searchResults.best_estimator_
    logger.info(f"DecisionTree model tuning with run_name: {run_name} done. ")
    logger.info(
        f"model features logged to mlfow with  experiment: DecisionTree-tune and run_name{run_name} and  done. ")

    if (save and save_path != None):
        helper.write_model(save_path, best_dt_Model)
    return best_dt_Model


if __name__ == "__main__":
    platform_df = get_data('enc-platform-df-v2')
    browser_df = get_data('enc-browser-df-v2')

    dt_clf_browser = helper.read_model(
        "../models/browser_decision_tree_model")
    dt_clf_platfrom = helper.read_model(
        "../models/platform_os_decision_tree_model")

    test_size = 0.1
    feature_browser_cols = ["experiment",
                            "hour", "date", 'device_make', 'browser']
    browser_X = browser_df[feature_browser_cols]
    browser_y = browser_df[['aware']]

    feature_platfrom_os_cols = ["experiment",
                                "hour", "date", 'device_make', 'platform_os']
    platfrom_X = platform_df[feature_platfrom_os_cols]
    platfrom_y = platform_df[['aware']]

    dt_browser_df_tuned = DecisionTree_tune(dt_clf_browser, split_train_test(browser_X, browser_y, test_size),
                                            loss_function=loss_function,
                                            run_name='DT-browser-model-Hyperparameter')

    dt_platfrom_df_tuned = DecisionTree_tune(dt_clf_platfrom, split_train_test(platfrom_X, platfrom_y, test_size),
                                             loss_function=loss_function,
                                             run_name='DT-platfrom-model-Hyperparameter')
