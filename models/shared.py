import pandas as pd
from sklearn.base import TransformerMixin


def load_data(train_path, test_path, y_col, id_col) -> tuple:
    """
    """
    y_col, id_col = y_col.lower(), id_col.lower()

    x_train = (pd.read_csv(train_path)
        .rename(columns=str.lower)
        .dropna(subset=y_col)
        .set_index(id_col)
    )
    y_train = x_train[y_col]

    x_train.drop([y_col], axis=1, inplace=True)
    x_test = (pd.read_csv(test_path)
        .set_index(id_col)
    )

    return x_train, y_train, x_test


class ConstantMapImputer(TransformerMixin):
    """ mimic simple imputer with constant, but allow
        for the constant to be dependent on value in `map_col`

        param: map_col str name of column that defines constant
        param: map dictionary that maps value in `map_col` to fill value 
    """
    def __init__(self, map_col, key_vals) -> None:
        self.map_col = map_col
        self.key_vals = key_vals

    def fit(self, X, y, **fit_params):
        self.fill_with = X[self.map_col].map(self.key_vals)
        return self

    def transform(self, X, **transform_params):
        return np.where(X.isna(), self.fill_with)



class SimpleOLSImputer(TransformerMixin):
    """
    """
    def __init__(self, x):
        self.model = LinearRegression()
        pass

    def fit(self, X, y, **fit_params):
        pass

    def tranform(self, X, **transform_params):
        # linreg = LinearRegression()
        # linreg.fit(X[x].values.reshape(-1, 1), x_train['lotfrontage'].values)
        # x_train['lotfrontage'].fillna(linreg.intercept_ + x_train['lotarea'] * linreg.coef_[0] , inplace=True)