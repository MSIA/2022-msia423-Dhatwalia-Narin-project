"""
This module defines the unit tests for train.py
"""
import warnings 
import pandas as pd
import numpy as np
import pytest

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
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
                           'trans_price'     :[30,
                                               165,
                                               145,
                                               155,
                                               40,
                                               50,
                                               132,
                                               154,
                                               35,
                                               120],
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

SEED = 2

# manually execute each step in the pipeline to obtain the expected model
OH_encoded_df = pd.get_dummies(original_df, drop_first=False)
x_train, x_test, y_train, y_test = train_test_split(OH_encoded_df.drop(['response'],axis=1),
                                                    OH_encoded_df['response'].values.ravel(),
                                                    test_size=0.50,
                                                    random_state=SEED)
scaler = StandardScaler()
scaled_df = scaler.fit_transform(x_train)
scaled_df = pd.DataFrame(scaled_df, index=x_train.index, columns=x_train.columns)
model = LogisticRegression(random_state=SEED,max_iter=15)
model.fit(scaled_df, y_train)

# define functions with happy paths
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
                                        random_state=SEED,
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
                                        random_state=SEED,
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
                                        random_state=SEED,
                                        max_iter=15)[0]
    actual_preds = output_model.predict_proba(x_test)[:, 1]
    print(actual_preds)
    expected_preds = model.predict_proba(x_test)[:, 1]
    print(expected_preds)
    assert list(actual_preds) == list(expected_preds)

def test_model_auc_score():
    """
    Check if the test set AUC score is the same as expected score
    """
    output_model = train.train_evaluate(features=OH_encoded_df.drop(['response'],axis=1),
                                        response=OH_encoded_df['response'].values.ravel(),
                                        results_path=None,
                                        matrix_path=None,
                                        roc_path=None,
                                        test_size=0.50,
                                        random_state=SEED,
                                        max_iter=15)[0]
    actual_preds = output_model.predict_proba(x_test)[:, 1]
    auc_actual = roc_auc_score(y_test, actual_preds)
    expected_preds = model.predict_proba(x_test)[:, 1]
    auc_expected = roc_auc_score(y_test, expected_preds)
    assert auc_actual == auc_expected

def test_model_log_loss():
    """
    Check if the test set log loss is the same as expected log loss
    """
    output_model = train.train_evaluate(features=OH_encoded_df.drop(['response'],axis=1),
                                        response=OH_encoded_df['response'].values.ravel(),
                                        results_path=None,
                                        matrix_path=None,
                                        roc_path=None,
                                        test_size=0.50,
                                        random_state=SEED,
                                        max_iter=15)[0]
    actual_preds = output_model.predict_proba(x_test)
    loss_actual =  log_loss(y_test, actual_preds)
    expected_preds = model.predict_proba(x_test)
    loss_expected = log_loss(y_test, expected_preds)
    assert loss_actual == loss_expected

def test_model_classification_report():
    """
    Check if the classification report is the same as expected report
    """
    output_model = train.train_evaluate(features=OH_encoded_df.drop(['response'],axis=1),
                                        response=OH_encoded_df['response'].values.ravel(),
                                        results_path=None,
                                        matrix_path=None,
                                        roc_path=None,
                                        test_size=0.50,
                                        random_state=SEED,
                                        max_iter=15)[0]
    actual_preds = output_model.predict(x_test)
    actual_report = classification_report(y_test, actual_preds,
                                                    output_dict=True,
                                                    zero_division=0)
    expected_preds = model.predict(x_test)
    expected_report = classification_report(y_test, expected_preds,
                                                    output_dict=True,
                                                    zero_division=0)
    assert actual_report == expected_report

# manually recreate the raw input transforms to obtain the expected model input

df_two_row = pd.DataFrame({'owner':['self','self'],
                           'ticker':['AAPL','GOOG'],
                           'type_trans':['purchase','sale_full'],
                           'amount':['$1,001 - $15,000','$1,001 - $15,000'],
                           'representative':['Hon. Alan S. Lowenthal','Hon. Kurt Schrader'],
                           'trans_price':[155.3, 124.5]})

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    enc = OneHotEncoder()
    dummy_categ = enc.fit_transform(df_two_row[['owner','ticker','type_trans','amount','representative']])
    dummy_categ = pd.DataFrame(dummy_categ.toarray())
    features = pd.concat([dummy_categ, df_two_row['trans_price']], axis=1)
    scaler = StandardScaler()
    scaler.fit_transform(features)
    cat_inputs = ['self','GOOG','sale_full','$1,001 - $15,000','Hon. Alan S. Lowenthal']
    correct_output = enc.transform([cat_inputs]).toarray()
    correct_output = np.append(correct_output[0], 153.6)
    correct_output = [correct_output]
    correct_output = scaler.transform(correct_output)

def test_transform():
    """
    Check if the transformed output is the same as expected
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        actual_output = train.transform(enc, scaler,cat_inputs, 153.6)
    print(actual_output)
    print(correct_output)
    assert list(actual_output[0]) == list(correct_output[0])
