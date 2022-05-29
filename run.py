"""
This is the calling script that leverages modules in the src folder
"""

import logging.config
import argparse
import os


import yaml

from src.createdb import (create_db,
                          add_df)
from src.acquire  import (get_file,
                          upload_s3,
                          download_s3)
from src.clean    import (clean_stockwatcher,
                          join_current_price,
                          join_transact_price,
                          add_response,
                          filter_df,
                          get_dummies,
                          drop_dups,
                          string_cleanup)

logging.config.fileConfig('config/logging/local.conf')

# Add parsers for both creating a database and uploading source data to s3 bucket
parser = argparse.ArgumentParser(description='Create database or upload data to s3')
parser.add_argument('--config', default='config/test.yaml',
                    help='Path to configuration file')

subparsers = parser.add_subparsers(dest='subparser_name')

# Sub-parser for creating a database
sb_create = subparsers.add_parser('create_db', description='Create database')

# Sub-parser for ingesting new data into s3 bucket
sb_ingest = subparsers.add_parser('acquire', description='Add data to s3 bucket')
sb_ingest.add_argument('--s3_raw',
                        required=False,
                        nargs = '+',
                        help='Will load data to specified path',
                        default='')

# Sub-parser for cleaning raw data from s3 bucket
sb_download = subparsers.add_parser('clean',
                                    description='Download & clean data from s3 bucket')                                 
sb_download.add_argument('--s3_raw',
                         required=False,
                         nargs = '+',
                         help='Will load data from specified path',
                         default='')

# Sub-parser for training and saving model
sb_train = subparsers.add_parser('train',
                                 description='Train model / OneHotEncoder and save to s3 bucket')

args = parser.parse_args()
sp_used = args.subparser_name


if __name__ == '__main__':
    with open(args.config, "r", encoding='utf8') as f:
        y_conf = yaml.load(f, Loader=yaml.FullLoader)

    if sp_used == 'acquire':
        # Download raw data from url
        get_file(**y_conf['acquire']['get_file']['recent_transactions'])
        get_file(**y_conf['acquire']['get_file']['stockwatcher'])
        get_file(**y_conf['acquire']['get_file']['current_price'])
        get_file(**y_conf['acquire']['get_file']['transact_price'])

        # Push the raw data to S3
        upload_s3(args.s3_raw[0], y_conf['acquire']['upload_s3']['recent_transactions'])
        upload_s3(args.s3_raw[1], y_conf['acquire']['upload_s3']['stockwatcher'])
        upload_s3(args.s3_raw[2], y_conf['acquire']['upload_s3']['current_price'])
        upload_s3(args.s3_raw[3], y_conf['acquire']['upload_s3']['transact_price'])

    elif sp_used == 'create_db':
        create_db()
        add_df(y_conf['create_db']['local_path'])

    elif sp_used == 'clean':
        download_s3(args.s3_raw[0], **y_conf['clean']['download_s3']['rt'])
        download_s3(args.s3_raw[1], **y_conf['clean']['download_s3']['sw'])
        download_s3(args.s3_raw[2], **y_conf['clean']['download_s3']['cp'])
        download_s3(args.s3_raw[3], **y_conf['clean']['download_s3']['tp'])

        data = clean_stockwatcher(**y_conf['clean']['stockwatcher'])
        data = join_transact_price(data, **y_conf['clean']['transact'])
        data = join_current_price(data, **y_conf['clean']['current'])
        data = add_response(data)
        data = filter_df(data, **y_conf['clean']['filter'])
        data = get_dummies(data)
        data = drop_dups(data)
        string_cleanup(data, **y_conf['clean']['s_cleanup'])

    else:
        parser.print_help()
