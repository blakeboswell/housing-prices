from datetime import datetime
from typing import Callable
import pandas as pd


import mlflow
from mlflow.tracking import MlflowClient

from sklearn.model_selection import RandomizedSearchCV


def output_path(filename: str) -> str:
    """
    """
    return f'/data/housing_prices/output/{filename}'


def create_submission(experiment_id: str, x: pd.DataFrame, best_estimator):
    """
    """
    preds = best_estimator.predict(x)
    filepath = output_path( f'submission_{experiment_id}.csv')
    (pd.DataFrame({'Id': x.index, 'SalePrice': preds})
       .to_csv(filepath, index=False))
    return filepath


def run_experiment(experiment_name: str, 
                   x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, 
                   pipeline, params: dict, 
                   score_fun: Callable, k_folds: int=5):
    """
    """
    client = MlflowClient()
    run_time = datetime.now().strftime('%Y%m%d%I%M%S')

    try:
        experiment_id = client.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=f'{experiment_name}_{run_time}'):
    
        grid = RandomizedSearchCV(
            pipeline, params, cv=k_folds, random_state=0, scoring=score_fun,    
            n_jobs = -1,
            error_score='raise'
        )
        grid.fit(x_train, y_train)
    
        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)

        filepath = create_submission(experiment_id, x_test, grid.best_estimator_)
        mlflow.log_artifact(filepath)

    return grid.best_estimator_