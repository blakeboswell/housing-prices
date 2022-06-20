import pandas as pd
from xgboost import XGBRegressor



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error



train_path = '/data/housing_prices/input/train.csv.gz'
test_path  = '/data/housing_prices/input/test.csv.gz'

# Read the data
train_data = pd.read_csv(train_path, index_col='Id')
test_data = pd.read_csv(test_path, index_col='Id')


# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data['SalePrice']            
train_data.drop(['SalePrice'], axis=1, inplace=True)

# get numeric columns and one-hot columns
dummies_cols = [x for x in train_data.columns 
                if train_data[x].nunique() < 10 and train_data[x].dtype == "object"]

numeric_cols = [x for x in train_data.columns 
                if train_data[x].dtype in ['int64', 'float64']]


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

# Define the model
model = XGBRegressor(random_state=0, n_estimators=200, learning_rate=0.05)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(clf, train_data, y, cv=5, scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())