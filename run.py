"""
This is the calling script that leverages modules in the src folder
"""

import logging.config
import argparse
import os

import yaml

from src.createdb import create_db, add_df
from src.acquire import get_file, upload_s3

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
sb_download.add_argument('--s3_raw', required=False,
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

    if sp_used == 'create_db':
        create_db()
        add_df(y_conf['create_db']['local_path'])

    elif sp_used == 'acquire':
        get_file(**y_conf['acquire']['get_file']['recent_transactions'])
        get_file(**y_conf['acquire']['get_file']['stockwatcher'])
        get_file(**y_conf['acquire']['get_file']['current_price'])
        get_file(**y_conf['acquire']['get_file']['transact_price'])
        print(args.s3_raw[0])
        upload_s3(args.s3_raw[0], y_conf['acquire']['upload_s3']['recent_transactions'])
        upload_s3(args.s3_raw[1], y_conf['acquire']['upload_s3']['stockwatcher'])
        upload_s3(args.s3_raw[2], y_conf['acquire']['upload_s3']['current_price'])
        upload_s3(args.s3_raw[3], y_conf['acquire']['upload_s3']['transact_price'])
    else:
        parser.print_help()
