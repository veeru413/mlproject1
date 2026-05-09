# here in data ingestion what we do is we read the data from the source and then we save it in the raw data 
# folder and then we return the path of the raw data folder so that we can use it in the
#  data transformation and model training part.
# also in companies they have this big data team which is responsible for data ingestion, 
# data transformation and data validation, so that they can keep the data in a good shape and 
# also they can keep the data in a good format for the model training part, 
# so that they can get good results from the model training part.

import os
import sys
from src.exception import CustomException # we are importing the CustomException class from the exception.py file that we have created in the src folder, so that we can use it to raise custom exceptions in our project.
from src.logger import logging # we are importing the logging module from the logger.py file that we have created in the src folder, so that we can log the information and error messages in our project.
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass # here we use decorator dataclass to create a dataclass for the data ingestion configuration, which will contain the path of the raw data folder, the path of the train data folder and the path of the test data folder.
class DataIngestionConfig: # we are creating a dataclass for the data ingestion configuration, which will contain the path of the raw data folder, the path of the train data folder and the path of the test data folder.
    train_data_path: str = os.path.join('artifacts','train.csv') # we are creating a path for the train data folder by joining the artifacts folder and the train.csv file name, so that we can save the train data in this path.
    test_data_path: str = os.path.join('artifacts','test.csv') # we are creating a path for the test data folder by joining the artifacts folder and the test.csv file name, so that we can save the test data in this path.
    raw_data_path: str = os.path.join('artifacts','data.csv') # we are creating a path for the raw data folder by joining the artifacts folder and the data.csv file name, so that we can save the raw data in this path.


class DataIngestion: # we are creating a class for the data ingestion, which will contain the method for the data ingestion, which will read the data from the source and then save it in the raw data folder and then return the path of the raw data folder so that we can use it in the data transformation and model training part.
    def __init__(self): # we are creating an __init__ method for the DataIngestion class, which will initialize the data ingestion configuration by creating an instance of the DataIngestionConfig class and assigning it to the data_ingestion_config attribute of the DataIngestion class.
        self.data_ingestion_config = DataIngestionConfig() # we are creating an instance of the DataIngestionConfig class and assigning it to the data_ingestion_config attribute of the DataIngestion class, so that we can use it to access the path of the raw data folder, the path of the train data folder and the path of the test data folder in our project.

    def initiate_data_ingestion(self): # we are creating a method for the data ingestion, which will read the data from the source and then save it in the raw data folder and then return the path of the raw data folder so that we can use it in the data transformation and model training part.
        logging.info("Entered the data ingestion method or component") # we are logging the information message that we have entered the data ingestion method or component, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
        try: # we are using try except block to catch any exception that occurs during the execution of the data ingestion method, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
            df = pd.read_csv('../../notebook/data/stud.csv') # we are reading the data from the source (notebook/data/stud.csv) using pandas read_csv method and assigning it to the df variable, so that we can use it to save it in the raw data folder and then return the path of the raw data folder. but we can also read from mongodb and all
            logging.info("Read the dataset as dataframe") # we are logging the information message that we have read the dataset as dataframe, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True) # we are creating the raw data folder if it does not exist, and also we are using exist_ok=True to avoid any error if the folder already exists, so that we can save the raw data in this folder.
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False) # we are saving the raw data in the raw data folder by using pandas to_csv method and passing the path of the raw data folder from the data ingestion configuration and also setting index=False to avoid saving the index column in the csv file, so that we can keep our csv file clean and organized.
            logging.info("Raw data is saved") # we are logging the information message that we have saved the raw data, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            
            logging.info("Train test split initiated") # we are logging the information message that we have initiated the train test split, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42) # we are splitting the data into train and test sets by using scikit-learn train_test_split method and passing the dataframe, test_size=0.2 to split 20% of the data as test set and random_state=42 to get the same split every time we run the code, so that we can keep our results consistent and reproducible.
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True) # we are saving the train and test data in the train data folder and test data folder respectively by using pandas to_csv method and passing the path of the train data folder and test data folder from the data ingestion configuration and also setting index=False to avoid saving the index column in the csv file and header=True to save the header in the csv file, so that we can keep our csv files clean and organized.
            logging.info("Ingestion of the data is completed") # we are logging the information message

            return (self.data_ingestion_config.train_data_path,
                    self.data_ingestion_config.test_data_path) # we are returning the path of the train data folder and test data folder from the data ingestion configuration, 
                    #so that we can use it in the data transformation and model training part, 
                    # so that we can get the train and test data for our model training part.


        except Exception as e: # if any exception occurs during the execution of the try block, then it will be caught here and assigned to the variable e, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
            logging.info("Error occurred in data ingestion component") # we are logging the information message that an error occurred in the data ingestion component, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            raise CustomException(e,sys) # we are raising a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file, so that we can get a custom error message with file name and line number where the exception occurred when we print the exception.


# So now so easily we did data ingestion part, we read the data from the source and then we saved it in the raw data folder and then we returned the path of the raw data folder so that we can use it in the data transformation and model training part,
# and also we have used logging to log the information messages and error messages in our project,

# testing
# if __name__ == "__main__":
#     obj = DataIngestion() # we are creating an object of the DataIngestion class, so that we can call the initiate_data_ingestion method to test the data ingestion part.
#     train_data,test_data = obj.initiate_data_ingestion() # we are calling the initiate_data_ingestion method of the DataIngestion class and assigning the returned path of the train data folder and test data folder to the train_data and test_data variables respectively, so that we can use it to check if the data ingestion part is working fine or not by printing the paths of the train data folder and test data folder.
#     print(train_data) # we are printing the path of the train data folder to check if it is correct or not.
#     print(test_data) # we are printing the path of the test data folder to check if it is correct or not.


#hence we get artifacts folder and the split
# hence we can read the data from any api or mongo db and then extract or ingest the data and then 
# save it in the raw data folder and then return the path of the raw data folder so that
#  we can use it in the data transformation and model training part,
