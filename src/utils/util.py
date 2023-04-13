import sys
import os
import numpy as np
import dill
from datetime import datetime
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


def save_numpy_array_data(file_path: str,  array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            np.save(file=file_obj, arr=array)

    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def save_object(file_path: str, obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(file=file_obj, obj=obj)

    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    file_path: str
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)
    
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    
    except Exception as e:
        logging.info(f"Error Occurred at {CustomException(e,sys)}")
        raise CustomException(e, sys)