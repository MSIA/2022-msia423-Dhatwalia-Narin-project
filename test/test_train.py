"""
This module defines the unit tests for train.py
"""
import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src import train


# create a sample DataFrame to mimic the cleaned data
original_df = pd.DataFrame({'owner'          :['dependent',
                                               'dependent',
                                               'self',
                                               'undisclosed',
                                               'dependent',
                                               'dependent',
                                               'self',
                                               'undisclosed',
                                               'dependent',
                                               'self'],
                           'ticker'          :['AAPL',
                                               'GOOG',
                                               'MSFT',
                                               'GOOG',
                                               'GOOG',
                                               'MSFT',
                                               'AAPL',
                                               'AAPL',
                                               'GOOG',
                                               'MSFT'],
                           'type'            :['purchase',
                                               'sale_full',
                                               'sale_full',
                                               'sale_partial',
                                               'purchase',
                                               'purchase',
                                               'sale_partial',
                                               'sale_full',
                                               'purchase',
                                               'sale_full'],
                           'amount'          :['$1,001 - $15,000',
                                               '$50,001 - $100,000',
                                               '$1,001 -',
                                               '$1,001 - $15,000',
                                               '$1,001 - $15,000',
                                               '$50,001 - $100,000',
                                               '$1,001 -',
                                               '$1,001 - $15,000',
                                               '$1,001 -',
                                               '$1,001 - $15,000'],
                           'representative'  :['Hon. Alan S. Lowenthal',
                                               'Hon. Alan S. Lowenthal',
                                               'Hon. Rohit Khanna',
                                               'Hon. Kurt Schrader',
                                               'Hon. Rohit Khanna',
                                               'Hon. Rohit Khanna',
                                               'Hon. Alan S. Lowenthal',
                                               'Hon. Kurt Schrader',
                                               'Hon. Alan S. Lowenthal',
                                               'Hon. Kurt Schrader'],
                           'trans_price'     :[150,
                                               165,
                                               145,
                                               155,
                                               170,
                                               110,
                                               152,
                                               134,
                                               170,
                                               110],
                           'response'        :[0,
                                               1,
                                               1,
                                               1,
                                               0,
                                               0,
                                               1,
                                               1,
                                               0,
                                               1]})

# manually execute each step in the pipeline to obtain the expected output
OH_encoded_df = pd.get_dummies(original_df, drop_first=False)
x_train, x_test, y_train, y_test = train_test_split(OH_encoded_df.drop(['response'],axis=1),
                                                    OH_encoded_df['response'].values.ravel(),
                                                    test_size=0.50,
                                                    random_state=10)
scaler = StandardScaler()
scaled_df = scaler.fit_transform(x_train)
scaled_df = pd.DataFrame(scaled_df, index=x_train.index, columns=x_train.columns)
model = LogisticRegression(random_state=10,max_iter=15)
model.fit(scaled_df, y_train)

# Define functions with happy paths
def test_model_coeffs():
    """
    Check if the logistic regression model learns the correct coefficients
    """
    output_model = train.train_evaluate(features=OH_encoded_df.drop(['response'],axis=1),
                                        response=OH_encoded_df['response'].values.ravel(),
                                        results_path=None,
                                        matrix_path=None,
                                        roc_path=None,
                                        test_size=0.50,
                                        random_state=10,
                                        max_iter=15)[0]
    actual_coeffs = [item for items in output_model.coef_.tolist() for item in items]
    expected_coeffs = [item for items in model.coef_.tolist() for item in items]
    assert actual_coeffs == expected_coeffs

def test_model_pred_classes():
    """
    Check if the test set class predictions are the same as expected predictions
    """
    output_model = train.train_evaluate(features=OH_encoded_df.drop(['response'],axis=1),
                                        response=OH_encoded_df['response'].values.ravel(),
                                        results_path=None,
                                        matrix_path=None,
                                        roc_path=None,
                                        test_size=0.50,
                                        random_state=10,
                                        max_iter=15)[0]
    actual_preds = output_model.predict(x_test)
    expected_preds = model.predict(x_test)
    assert list(actual_preds) == list(expected_preds)

def test_model_pred_probs():
    """
    Check if the test set probabilty predictions are the same as expected predictions
    """
    output_model = train.train_evaluate(features=OH_encoded_df.drop(['response'],axis=1),
                                        response=OH_encoded_df['response'].values.ravel(),
                                        results_path=None,
                                        matrix_path=None,
                                        roc_path=None,
                                        test_size=0.50,
                                        random_state=10,
                                        max_iter=15)[0]
    actual_preds = output_model.predict_proba(x_test)[:, 1]
    print(actual_preds)
    expected_preds = model.predict_proba(x_test)[:, 1]
    print(expected_preds)
    assert list(actual_preds) == list(expected_preds)

def test_model_AUC_score():
    """
    Check if the test set probabilty predictions are the same as expected predictions
    """
    output_model = train.train_evaluate(features=OH_encoded_df.drop(['response'],axis=1),
                                        response=OH_encoded_df['response'].values.ravel(),
                                        results_path=None,
                                        matrix_path=None,
                                        roc_path=None,
                                        test_size=0.50,
                                        random_state=10,
                                        max_iter=15)[0]
    actual_preds = output_model.predict_proba(x_test)[:, 1]
    print(actual_preds)
    expected_preds = model.predict_proba(x_test)[:, 1]
    print(expected_preds)
    assert list(actual_preds) == list(expected_preds)