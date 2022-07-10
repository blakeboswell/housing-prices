import pandas as pd
from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


train_path = '/data/housing_prices/input/train.csv.gz'
test_path  = '/data/housing_prices/input/test.csv.gz'

# Read the data
raw_data   = pd.read_csv(train_path, index_col='Id')
score_data = pd.read_csv(test_path, index_col='Id')


# Remove rows with missing target, separate target from predictors
raw_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = raw_data['SalePrice']
raw_data.drop(['SalePrice'], axis=1, inplace=True)
X_train, y_train, X_valid, y_valid = train_test_split(raw_data, y, random_state=0)


# get numeric columns and one-hot columns
dummies_cols = [x for x in X_train.columns 
                if X_train[x].nunique() < 10 and X_train[x].dtype == "object"]

numeric_cols = [x for x in X_train.columns 
                if X_train[x].dtype in ['int64', 'float64']]


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, dummies_cols)
    ])


def get_score(n_estimators: int) -> float:
    """Return the average MAE over 3 CV folds of xgboost
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('model', XGBRegressor(n_estimators=n_estimators, random_state=0, learning_rate=0.01))]
    )
    
    scores = -1 * cross_val_score(clf, X_train, y, cv=3, scoring='neg_mean_absolute_error')
    
    return scores.mean()



# results = {x: get_score(x) for x in range(450, 850, 50)}

# for k, v in results.items():
#     print(k, v)

# test output

min_child_weight = 10
gamma = 0.5
colsample_bytree = 0.6
max_depth = 5
# learning_rate=0.01,

model = XGBRegressor(
    n_estimators=600, random_state=0, 
    min_child_weight=min_child_weight, gamma=gamma, colsample_bytree=colsample_bytree, max_depth=max_depth,
)
clf = Pipeline(
    steps=[('preprocessor', preprocessor),
            ('model', model)]
)
clf.fit(X_train, y)
preds = clf.predict(test_data)

output = pd.DataFrame({'Id': test_data.index,
                       'SalePrice': preds})
output.to_csv('/data/housing_prices/output/submission2.csv', index=False)

