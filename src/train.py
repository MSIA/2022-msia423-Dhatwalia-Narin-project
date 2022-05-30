"""
This module trains and evaluates a logistic regression model for binary classification
"""
import logging
import pickle

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

def train(local_path,
          categ,
          response,
          results_path,
          model_path,
          encoder_path,
          test_size,
          random_state,
          max_iter):
    '''Orchestration function to train, evaluate & save profitability prediction model
    Args:
        local_path (str): path to cleaned data
        categ (list): list of column names representing categorical features
        response (str): column representing response
        results_path (str): path to write model evaluation results to
        model_path (str): path to pickled model
        encoder_path (str): path to pickled encoder
        test_size (float): fraction of original data to split into test set
        random_state (int): random state for training model
        max_iter (int): Maximum number of iterations taken for the solvers to converge.
    Returns:
        None
    '''
    try:
        data = pd.read_csv(local_path)
    except FileNotFoundError:
        logger.error("File %s not found at ", local_path)
        logger.debug("Check path in the configuration file")

    enc = OneHotEncoder().fit(data[categ])  # fit to categorical vars
    dummy_categ = enc.transform(data[categ])
    dummy_categ = pd.DataFrame(dummy_categ.toarray())

    # concatenate categorical features and numeric features
    features = pd.concat([dummy_categ, data.drop(categ+[response], axis=1)], axis=1)
    features.columns = [str(i) for i in features.columns]

    #response = np.array(data[response])
    response = data[response].values.ravel()

    model = train_evaluate(features, response, results_path, test_size,
                           random_state, max_iter)
    pickle.dump(model, open(model_path, 'wb'))
    logger.info("Model saved to: %s", model_path)
    pickle.dump(enc, open(encoder_path, 'wb'))
    logger.info("OneHotEncoder saved to: %s", encoder_path)

def train_evaluate(features, response, results_path, test_size, random_state,
                   max_iter):
    '''Function to split data, train model, and evaluate model
    Args:
        features (pandas.core.frame.DataFrame): DataFrame holding feature variables
        response (numpy.ndarray): array holding responses for each individual
        results_path (str): path to write model evaluation results to
        test_size (float): fraction of original data to split into test set
        random_state (int): random state for training model
        max_depth (int): max depth of trees in random forest model
        n_estimators (int): number of trees in random forest model
    Returns:
        ovr (sklearn.multiclass.OneVsRestClassifier): multilabel random forest model
    '''
    x_train, x_test, y_train, y_test = train_test_split(
                                                        features, response,
                                                        test_size=test_size,
                                                        random_state=random_state)
    model = LogisticRegression(max_iter=max_iter,
                               random_state=random_state)
    logger.debug("Model training")
    ovr = model.fit(x_train.values, y_train)
    ypred_bin_test = ovr.predict(x_test.values)
    ypred_proba_test = ovr.predict_proba(x_test)

    auc = roc_auc_score(y_test, ypred_bin_test)
    loss = log_loss(y_test, ypred_proba_test)
    creport = classification_report(y_test, ypred_bin_test,
                                                    output_dict=True)

    results = [creport, {"AUC": str(auc), "Log Loss": str(loss)}]
    with open(results_path, 'w',encoding='utf8') as file:
        outdoc = yaml.dump(results, file)
    logger.info("Model results written to: %s", results_path)

    return ovr

def get_model(model_path, encoder_path):
    '''Opens pickled model and encoder for the data
    Args:
        model_path (str): path to pickled model
        encoder_path (str): path to pickled encoder
    Returns:
        model (): binary classifier logistic regression model
        encoder (sklearn.preprocessing._encoders.OneHotEncoder): encoder for categorical variables
    '''
    try:
        with open(model_path, "rb") as input_file:
            model = pickle.load(input_file)
    except FileNotFoundError:
        logger.error("File %s not found at ", model_path)
        logger.debug("Check path in the configuration file")
    except Exception as error:
        logger.error("General error reading file: %s", error)
        logger.debug("Check file location for: %s", model_path)
    try:
        with open(encoder_path, "rb") as input_file:
            enc = pickle.load(input_file)
    except FileNotFoundError:
        logger.error("File %s not found at ", encoder_path)
        logger.debug("Check path in the configuration file")
    except Exception as error:
        logger.error("General error reading file: %s", error)
        logger.debug("Check file location for: %s", encoder_path)

    return model, enc

def transform(encoder, cat_inputs, trans_price):
    '''Transforms raw input into encoded input for model use
    Args:
        encoder (sklearn.preprocessing._encoders.OneHotEncoder): encoder for categorical variables
        cat_inputs (:obj:`list` of :obj:`str`): categorical inputs of individual
        trans_price (float): stock price on the day of transaction
    Returns:
        test_new (2D :obj:`list` of :obj:`int): encoded inputs for model prediction
    '''
    test_new = encoder.transform([cat_inputs]).toarray()  # needs 2d array
    test_new = np.append(test_new[0], trans_price)  # encoder returns 2d array, need element inside
    test_new = [test_new]  # predict function expects 2d arrray
    return test_new

def predict_ind(model, encoder, cat_inputs, trans_price):
    '''Predicts the probabilities for a new model
    Args:
        model (sklearn.multiclass.OneVsRestClassifier): binary logistic regression model
        encoder (sklearn.preprocessing._encoders.OneHotEncoder): encoder for categorical variables
        cat_inputs (list): categorical inputs of individual
        trans_price (float): birth year for individual
    Returns:
        prediction (numpy.ndarray): probability of profitable investment
    '''
    test_new = transform(encoder, cat_inputs, trans_price)
    prediction = model.predict_proba(test_new)
    #prediction = prediction[1]
    prediction = prediction[0][1]
    return prediction
