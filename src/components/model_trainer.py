import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils.util import evaluate_model, load_numpy_array_data, save_object, get_current_time_stamp


class ModelTrainerConfig:
    taine_model_file_path = os.path.join(
        'artifact', 'model_trainer', get_current_time_stamp(), 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(Self, train_array_path, test_array_path):

        try:
            train_array = load_numpy_array_data(train_array_path)
            test_array = load_numpy_array_data(test_array_path)

            logging.info(
                'Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()}
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
            

        except Exception as e:
            logging.info(f"Error Occurred at {CustomException(e,sys)}")
            raise CustomException(e, sys)
