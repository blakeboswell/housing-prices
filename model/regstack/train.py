import sys
sys.path.append('../..')

from yaml import load
from preprocess.load import load_data
from preprocess.baseline import baseline_trans


import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient


from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.compose import make_column_transformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict


def rmsle(y, y0):
    """
    """
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


def tune_parameters(x_train, y_train, preprocessor, k_folds=5) -> tuple:
    """
    """

    cat_selector = make_column_selector(dtype_include=object)
    num_selector = make_column_selector(dtype_include=np.number)

    tree_preprocessor = Pipeline(steps=[
        ('baseline', preprocessor),
        ('jic', SimpleImputer(strategy='most_frequent', add_indicator=True))
    ])
    num_linear_processor = make_pipeline(
        StandardScaler(), 
        SimpleImputer(strategy='mean', add_indicator=True)
    )
    linear_preprocessor = make_column_transformer(
        (num_linear_processor, num_selector),
        (OneHotEncoder(handle_unknown='ignore'), cat_selector)
    )
    lasso_pipeline = make_pipeline(linear_preprocessor, LassoCV())
    rf_pipeline = make_pipeline(tree_preprocessor, RandomForestRegressor(random_state=42))

    gbdt_pipeline = make_pipeline(
        tree_preprocessor, HistGradientBoostingRegressor(random_state=0)
    )
    
    estimators = [
        ('Random Forest', rf_pipeline),
        ('Lasso', lasso_pipeline),
        ('Gradient Boosting', gbdt_pipeline),
    ]

    stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())

    rmsle_score = make_scorer(rmsle, greater_is_better=False)
    kfold_scores = cross_val_score(
        stacking_regressor, x_train, y_train, scoring=rmsle_score, verbose=0
    )

    mlflow.log_metric(f"average_accuracy", kfold_scores.mean())
    mlflow.log_metric(f"std_accuracy", kfold_scores.std())

    return stacking_regressor.fit(x_train, y_train)



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
    data_name = 'regstack'
    
    try:
        experiment_id = client.create_experiment(data_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = client.get_experiment_by_name(data_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='test'):
        # mlflow.sklearn.autolog()

        x_train, y_train, x_test = load_data()
        best_estimator = tune_parameters(x_train, y_train, preprocessor=baseline_trans)
        filepath = create_submission(experiment_id, x_test, best_estimator)
        mlflow.log_artifact(filepath)

