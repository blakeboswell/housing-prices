import numpy as np
import pandas as pd

train_path = '/data/housing_prices/input/train.csv.gz'
test_path  = '/data/housing_prices/input/test.csv.gz'
y_col      = 'SalePrice'
id_col     = 'Id'


def load_data() -> tuple:
    """ load x_train, y_train, and x_test
    """
    x_train = (pd.read_csv(train_path)
                 .dropna(subset=[y_col])
                 .set_index([id_col])
    )
    y_train = x_train[y_col]
    x_train.drop([y_col], axis=1, inplace=True)

    x_test = (pd.read_csv(test_path)
                .set_index([id_col])
    )

    return x_train, y_train, x_test
    # return x_train, np.log(y_train), x_test