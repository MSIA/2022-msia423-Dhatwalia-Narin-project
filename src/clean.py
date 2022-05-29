"""
This module cleans the raw data and prepares it for modeling
"""
import logging
from operator import index
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Function 1
def clean_stockwatcher(local_path):
    '''Takes raw data and cleans it for model use
    Args:
        local_path (str): path to raw data file
        save_path (str): path to save the cleaned file
    Returns:
        None
    '''
    try:
        data = pd.read_csv(local_path)
    except FileNotFoundError:
        logger.error("File %s not found", local_path)
        logger.debug("Check path in the configuration file")

    data = data.drop(['Unnamed: 0'], axis=1)
    names = data['representative'].value_counts().head(15)
    names = list(names.index)
    names.remove('Hon. Donna Shalala')
    names.append("Hon. Nancy Pelosi")

    ticks = data['ticker'].value_counts().head(11)
    ticks = list(ticks.index)
    ticks.remove('--')

    data = data[data['representative'].isin(names)]
    data = data[data['ticker'].isin(ticks)]

    return data

# Function 2
def join_transact_price(data, local_path):
    """
    Insert docstring here
    """
    try:
        data_2 = pd.read_csv(local_path)
    except FileNotFoundError:
        logger.error("File not found")
        logger.debug("Check path in the configuration file")

    transact_price = data_2.drop_duplicates()
    transact_price = transact_price[['ticker','date','price']]

    #Add transaction day stock-price
    new_df = pd.merge(data,
                      transact_price[~np.isnan(transact_price["price"])],
                      how='inner',
                      left_on=['ticker','transaction_date'],
                      right_on = ['ticker','date'])

    new_df = new_df.rename(columns={'price': 'trans_price'})
    return new_df

# Function 3
def join_current_price(data, local_path):
    """
    Insert docstring here
    """
    try:
        data_2 = pd.read_csv(local_path)
    except FileNotFoundError:
        logger.error("File not found")
        logger.debug("Check path in the configuration file")

    current_price = data_2.drop_duplicates()
    current_price = current_price[['ticker','date','price']]
    current_price['date'] = current_price['date'].replace(['2022-04-06'],'2022-04-07')

    #Add current stock-price
    new_df = pd.merge(data,
                      current_price[~np.isnan(current_price["price"])],
                      how='inner',
                      left_on=['ticker'],
                      right_on = ['ticker'])

    new_df = new_df.rename(columns={'price': 'current_price'})
    return new_df

# Function 4
def add_response(data):
    """
    Insert docstring here
    """

    #Add binary variable for response
    data['response'] = data['current_price'] > data['trans_price']

    #Change the type from boolean to int
    data['response'] = data['response']*1
    return data

# Function 5
def filter_df(data, columns):
    '''
    Insert docstring here
    '''
    new_df = data.drop(columns=columns)
    return new_df

# Function 6
def get_dummies(data, drop_first = True):
    """
    Insert docstring here
    """
    new_df = pd.get_dummies(data, drop_first=drop_first)
    return new_df

# Function 7
def drop_dups(data):
    """
    Insert docstring here
    """
    new_df = data.drop_duplicates()
    return new_df

# Function 8
def string_cleanup(new_df, save_path):
    """
    Insert docstring here
    """
    new_df.columns = new_df.columns.str.replace("representative_Hon. ", "", regex=False)
    new_df.columns = new_df.columns.str.replace("representative_Mr. ", "",regex=False)

    new_df.columns = new_df.columns.str.replace("$1,001 - $15,000", "2",regex=False)
    new_df.columns = new_df.columns.str.replace("$1,001 -", "1",regex=False)
    new_df.columns = new_df.columns.str.replace("$100,001 - $250,000", "3",regex=False)
    new_df.columns = new_df.columns.str.replace("$15,001 - $50,000", "4",regex=False)
    new_df.columns = new_df.columns.str.replace("$250,001 - $500,000", "5",regex=False)
    new_df.columns = new_df.columns.str.replace("$5,000,001 - $25,000,000", "6",regex=False)
    new_df.columns = new_df.columns.str.replace("$50,001 - $100,000", "7",regex=False)
    new_df.columns = new_df.columns.str.replace("$500,001 - $1,000,000", "8",regex=False)

    new_df.columns = new_df.columns.str.replace(".", "",regex=False)
    new_df.columns = new_df.columns.str.replace(" ", "_",regex=False)

    new_df.to_csv(save_path, index = False)
