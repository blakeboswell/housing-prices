import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


train_path = '/data/housing_prices/input/train.csv.gz'
test_path  = '/data/housing_prices/input/test.csv.gz'


def rmsle(y, y0):
    """
    """
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


def load_data(train_path, test_path, y_col, id_col) -> tuple:
    """
    """
    x_train = (pd.read_csv(train_path)
        .rename(columns=str.lower)
        .dropna(subset=y_col)
        .set_index(id_col)
    )
    y_train = x_train[y_col]
    x_train.drop([y_col], axis=1, inplace=True)
    x_test = (pd.read_csv(test_path)
        .rename(columns=str.lower)
        .set_index(id_col)
    )

    return x_train, y_train, x_test


def trans_missing(x_train: pd.DataFrame) -> ColumnTransformer:
    """
    """
    dummies_cols = [
        x for x in x_train.columns 
        if x_train[x].nunique() < 10 and x_train[x].dtype == "object"]

    numeric_cols = [
        x for x in x_train.columns 
        if x_train[x].dtype in ['int64', 'float64']]

    numerical_transformer = SimpleImputer(strategy='median')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_cols),
            ('cat', categorical_transformer, dummies_cols)
        ])
    
    # none_tran = Pipeline(steps=[
    #         ('impute', SimpleImputer(strategy='constant', fill_value='None')),
    #         ('onehot', OneHotEncoder(handle_unknown='ignore'))
    #     ])
    # none_cols = [
    #     'alley', 'masvnrtype', 'garagetype', 'miscfeature'
    #     , 'bsmtqual', 'bsmtcond', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2'
    #     , 'fireplacequ'
    #     , 'garagefinish', 'garagequal', 'garagecond'
    #     , 'poolqc'
    #     , 'fence'
    # ]

    # mode_tran = Pipeline(steps=[
    #         ('impouter', SimpleImputer(strategy='most_frequent')),
    #         ('onehot', OneHotEncoder(handle_unknown='ignore'))
    #     ])
    # mode_cols = ['electrical']

    # zero_tran = SimpleImputer(strategy='constant', fill_value=0.0)
    # zero_cols = ['masvnrarea', 'garageyrblt']

    # median_tran = SimpleImputer(strategy='most_frequent')
    # median_cols = ['lotfrontage']

    # return ColumnTransformer(
    #     transformers=[
    #         ('none', none_tran, none_cols),
    #         ('zero', zero_tran, zero_cols),
    #         ('cat', mode_tran, mode_cols),
    #         ('median', median_tran, median_cols)
    #     ])


def trans_prep():
    """
    """
    pass


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
    mlflow.sklearn.autolog()
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
    mlflow.log_metric(f"average_accuracy", kfold_scores.mean())
    mlflow.log_metric(f"std_accuracy", kfold_scores.std())
    

def create_submission(data_name, x, best_estimator):
    """
    """
    preds = best_estimator.predict(x)
    output = pd.DataFrame({'Id': x.index, 'SalePrice': preds})
    output.to_csv(f'/data/housing_prices/output/{data_name}_submission.csv', index=False)
    mlflow.log_artifact(f'/data/housing_prices/output/{data_name}_submission.csv')

if __name__ == '__main__':

    client = MlflowClient()
    data_name = 'test'
    
    try:
        experiment_id = client.create_experiment(data_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = client.get_experiment_by_name(data_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='test'):
        
        x_train, y_train, x_test = load_data(train_path, test_path, 'saleprice', 'id')
        preprocessor = trans_missing(x_train)
        best_estimator = tune_parameters(x_train, y_train, preprocessor)

        validation(x_train, y_train, best_estimator)
        create_submission(data_name, x_test, best_estimator)

