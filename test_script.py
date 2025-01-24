import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def test_dataset_availability():
    assert os.path.exists('dataset/train.csv'), "Training dataset is missing"
    assert os.path.exists('dataset/test.csv'), "Test dataset is missing"

def test_dataset_columns():
    train_data = pd.read_csv('dataset/train.csv')
    expected_columns = ['SalePrice', 'OverallQual', 'GrLivArea']  # Add expected column names
    for column in expected_columns:
        assert column in train_data.columns, f"Missing expected column: {column}"

def test_no_missing_target():
    train_data = pd.read_csv('dataset/train.csv')
    assert train_data['SalePrice'].notna().all(), "Target column contains missing values"


def test_one_hot_encoding():
    train_data = pd.read_csv('dataset/train.csv', index_col='Id')
    test_data = pd.read_csv('dataset/test.csv', index_col='Id')
    
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data['SalePrice']
    train_data.drop(['SalePrice'], axis=1, inplace=True)

    X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2, random_state=0)
    
    low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                            X_train[cname].dtype == "object"]
    numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
    
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train[my_cols].copy()
    X_valid = X_valid[my_cols].copy()
    X_test = test_data[my_cols].copy()
    
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)
    
    assert X_train.shape[1] == X_valid.shape[1], "Mismatch in number of columns after one-hot encoding (train vs. valid)"
    assert X_train.shape[1] == X_test.shape[1], "Mismatch in number of columns after one-hot encoding (train vs. test)"


def test_model_training_and_prediction():
    train_data = pd.read_csv('dataset/train.csv', index_col='Id')
    test_data = pd.read_csv('dataset/test.csv', index_col='Id')
    
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data['SalePrice']
    train_data.drop(['SalePrice'], axis=1, inplace=True)
    
    X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2, random_state=0)
    
    low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                            X_train[cname].dtype == "object"]
    numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
    
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train[my_cols].copy()
    X_valid = X_valid[my_cols].copy()
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    
    model = XGBRegressor(random_state=0)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_valid)
    mae = mean_absolute_error(predictions, y_valid)
    assert mae > 0, f"Model evaluation failed: MAE is {mae}"

def test_mae_threshold():
    train_data = pd.read_csv('dataset/train.csv', index_col='Id')
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data['SalePrice']
    train_data.drop(['SalePrice'], axis=1, inplace=True)
    
    X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2, random_state=0)
    
    low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 
                            X_train[cname].dtype == "object"]
    numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
    
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train[my_cols].copy()
    X_valid = X_valid[my_cols].copy()
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    
    model = XGBRegressor(random_state=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    mae = mean_absolute_error(predictions, y_valid)
    max_mae_threshold = 20000  # Set a threshold value based on domain knowledge
    assert mae <= max_mae_threshold, f"MAE exceeds the threshold: {mae}"
