import os
import sys
import pandas as pd
import numpy as np
from src.util import get_current_time_stamp
from src.exception import CustomException
from src.logger import logging
import urllib
from builtins import open
import requests
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join(
        'artifact', 'data_ingestion', get_current_time_stamp(), 'train_data', 'train.csv')
    test_data_path = os.path.join(
        'artifact', 'data_ingestion', get_current_time_stamp(), 'test_data', 'test.csv')
    raw_data_path = os.path.join(
        'artifact', 'data_ingestion', get_current_time_stamp(), 'raw_data', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(f"\n\n{'='*20} Data Started {'='*20}\n\n")

        try:
            df = pd.read_csv(
                'https://raw.githubusercontent.com/krishnaik06/FSDSRegression/main/notebooks/data/gemstone.csv')
            logging.info('Dataset Read Successfully')

            os.makedirs(os.path.dirname(
                self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            logging.info('Raw Data Saved as csv and train test split starting')
            train_set, test_set = train_test_split(df, test_size=0.3)

            os.makedirs(os.path.dirname(
                self.data_ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(
                self.data_ingestion_config.test_data_path), exist_ok=True)
            train_set.to_csv(
                self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(
                self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info('Saved training and testing data')
            logging.info(f"\n\n{'='*20} Data Ingestion Log Completed {'='*20} \n\n")

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
