"""
This module cleans the raw data and prepares it for modeling
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# function to join the transaction data with historical stock price data
def join_transact_price(input_path_1:str,
                        input_path_2:str) -> pd.core.frame.DataFrame:
    """
    This function joins the transaction data with the transaction-day stock
    price data by inner-joining the two DataFrames on the date and ticker columns

    Args:
        input_path_1 (str): path to the transaction data
        input_path_2 (str): path to the stock price data
    Returns:
        new_df (pd.core.frame.DataFrame): new DataFrame with transactions and stock price
        on the day of the transaction
    """
    try:
        data = pd.read_csv(input_path_1)
        data_2 = pd.read_csv(input_path_2)
    except FileNotFoundError:
        logger.error('File not found')
        logger.debug('Check path in the configuration file')

    transact_price = data_2.drop_duplicates()
    transact_price = transact_price[['ticker','date','price']]

    # add transaction day stock-price
    new_df = pd.merge(data,
                      transact_price[~np.isnan(transact_price['price'])],
                      how='inner',
                      left_on=['ticker','transaction_date'],
                      right_on = ['ticker','date'])

    new_df = new_df.rename(columns={'price': 'trans_price'})
    logger.info('Join #1 of the DataFrames completed successfully')
    return new_df

# function to join the transaction data with current stock price data
def join_current_price(data: pd.core.frame.DataFrame,
                       local_path: str) -> pd.core.frame.DataFrame:
    """
    This function joins the current price data with the transactions data
    by inner-joining the two DataFrames on the date and ticker columns

    Args:
        data (pd.core.frame.DataFrame): DataFrame containing transactions and historical prices
        local_path (str): path to the current stock price data
    Returns:
        new_df (pd.core.frame.DataFrame): new DataFrame with transactions and stock price
        on the day of the transaction
    """
    try:
        data_2 = pd.read_csv(local_path)
    except FileNotFoundError:
        logger.error('File not found')
        logger.debug('Check path in the configuration file')

    current_price = data_2.drop_duplicates()
    current_price = current_price[['ticker','date','price']]
    current_price['date'] = current_price['date'].replace(['2022-04-06'],'2022-04-07')

    # add current stock-price
    new_df = pd.merge(data,
                      current_price[~np.isnan(current_price['price'])],
                      how='inner',
                      left_on=['ticker'],
                      right_on = ['ticker'])

    new_df = new_df.rename(columns={'price': 'current_price'})
    logger.info('Join #2 of the DataFrames completed successfully')
    return new_df

# function to create the response variable
def add_response(data:pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    This function creates the response variable by comparing historical
    stock price and current stock price. It assigns a value of 1 is the
    current price is higher than the historical price, else it assigns 0

    Args:
        data (pd.core.frame.DataFrame): DataFrame without the response variable
    Returns:
        data (pd.core.frame.DataFrame): DataFrame with the response included
    """

    #Add binary variable for response
    data['response'] = data['current_price'] > data['trans_price']

    #Change the type from boolean to int
    data['response'] = data['response']*1
    logger.info('Response variable added successfully')
    return data

# function to exclude certain variables
def filter_df(data, columns):
    '''
    Insert docstring here
    '''
    new_df = data.drop(columns=columns)
    logger.info('DataFrame filtered successfully')
    return new_df

# Function 6
def drop_dups(data):
    """
    Insert docstring here
    """
    new_df = data.drop_duplicates()
    records_droped = data.shape[0] - new_df.shape[0]
    logger.info('Duplicates dropped = %s', records_droped)
    return new_df

# Function 7
def impute_missing(data,
                   save_path,
                   column = 'owner',
                   replacement = 'undisclosed',
                   missing_val = '--'):
    """
    Add docstring here
    """

    data[column] = data[column].fillna(replacement)
    data[column] = data[column].replace({missing_val:replacement})

    data.to_csv(save_path, index = False)
    logger.info('Cleaned data saved to %s', save_path)
