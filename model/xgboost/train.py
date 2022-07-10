import sys
sys.path.append('../..')

from yaml import load
from preprocess.load import load_data
from preprocess.baseline import baseline_trans


import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV



def rmsle(y, y0):
    """
    """
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


def tune_parameters(x_train, y_train, preprocessor, k_folds=5) -> tuple:
    """
    """
    params = {
        'model__min_child_weight': [1, 5, 8, 10],
        'model__gamma': [0.5, 1, 1.5, 2, 5],
        'model__subsample': [0.4, 0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'model__max_depth': [3, 4, 5, 8, 10]
    }

    rmsle_score = make_scorer(rmsle, greater_is_better=False)

    pipe = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('model', XGBRegressor(n_estimators=600, tree_method='gpu_hist', gpu_id=0, random_state=0))]
    )
    grid = RandomizedSearchCV(pipe, params, cv=k_folds, random_state=0, scoring=rmsle_score)
    grid.fit(x_train, y_train)
    
    for k, v in grid.best_params_.items():
        mlflow.log_param(k, v)

    return grid.best_estimator_


def validation(x, y, best_estimator, k_folds=5) -> None:
    """
    """
    rmsle_score = make_scorer(rmsle, greater_is_better=False)
    kfold_scores = cross_val_score(best_estimator, x, y, cv=k_folds, scoring=rmsle_score)  
    # mlflow.log_metric(f"average_accuracy", kfold_scores.mean())
    # mlflow.log_metric(f"std_accuracy", kfold_scores.std())
    

def create_submission(experiment_id, x, best_estimator):
    """
    """
    preds = best_estimator.predict(x)
    output = pd.DataFrame({'Id': x.index, 'SalePrice': preds})
    filepath = f'/data/housing_prices/output/submission_{experiment_id}.csv'
    output.to_csv(filepath, index=False)
    return filepath


if __name__ == '__main__':

    client = MlflowClient()
    data_name = 'test'
    
    try:
        experiment_id = client.create_experiment(data_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = client.get_experiment_by_name(data_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='test'):
        # mlflow.sklearn.autolog()

        x_train, y_train, x_test = load_data()
        best_estimator = tune_parameters(x_train, y_train, preprocessor=baseline_trans)

        validation(x_train, y_train, best_estimator)
        
        filepath = create_submission(experiment_id, x_test, best_estimator)
        mlflow.log_artifact(filepath)

