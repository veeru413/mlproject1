# the use of data tranformation is for data cleaning, feature engineering and feature selection, 
# so that we can keep our data in a good shape and also we can keep our data in a good format for the model
#  training part, so that we can get good results from the model training part.
# also converting categorical data into numerical data, so that we can use it in the model training part,
# and also we can use it for feature engineering, so that we can create new features from the existing 
# features, and also we can use it for feature selection, so that we can select the important features 
# from the existing features, so that we can keep our data in a good shape and also we can keep our data 
# in a good format for the model training part, so that we can get good results from the model training part.

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # column transformer is used to create the pipeline for the 
#data transformation, so that we can apply different transformations to different columns of the data, 
# so that we can keep our data in a good shape and also we can keep our data in a good format for the 
# model training part, so that we can get good results from the model training part.

from sklearn.impute import SimpleImputer # simple imputer is used to fill the missing values in the data, 
#so that we can keep our data in a good shape and also we can keep our data in a good format for the model 
# training part, so that we can get good results from the model training part.

from sklearn.pipeline import Pipeline # pipeline is used to create a pipeline for the data transformation,
#so that we can apply different transformations to different columns of the data, so that we can keep our 
# data in a good shape and also we can keep our data in a good format for the model training part, 
# so that we can get good results from the model training part.

from sklearn.preprocessing import OneHotEncoder,StandardScaler 
from src.exception import CustomException # we are importing the CustomException class from the exception.py file that we have created in the src folder, so that we can use it to raise custom exceptions in our project.
from src.logger import logging # we are importing the logging module from the logger.py file that we have created in the src folder, so that we can log the information and error messages in our project.
from src.utils import save_object # we are importing the save_object function from the utils.py file that we have created in the src folder, so that we can use it to save the preprocessor object in pickle format.

@dataclass # here we use decorator dataclass to create a dataclass for the data transformation configuration, which will contain the path of the preprocessor object file, so that we can save the preprocessor object in this path and then we can use it in the model training part to transform the data before training the model.
class DataTransformationConfig: # we are creating a class for the data transformation configuration, which will contain the path of the preprocessor object file, so that we can save the preprocessor object in this path and then we can use it in the model training part to transform the data before training the model.
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl') # we are creating a path for the preprocessor object file by joining the artifacts folder and the preprocessor.pkl file name, so that we can save the preprocessor object in this path and then we can use it in the model training part to transform the data before training the model.


class DataTransformation: # we are creating a class for the data transformation, which will contain the method for the data transformation, which will create a pipeline for the data transformation and then save the preprocessor object in the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.
    def __init__(self): # we are creating an __init__ method for the DataTransformation class, which will initialize the data transformation configuration by creating an instance of the DataTransformationConfig class and assigning
        self.data_transformation_config = DataTransformationConfig() # we are creating an instance of the DataTransformationConfig class and assigning

    def get_data_transformer_object(self): # we are creating a method for the data transformation, which will create a pipeline for the data transformation and then return the preprocessor object, so that we can use it in the model training part to transform the data before training the model.
        '''
        this function is responsible for data transformation, which means that it will create a 
        pipeline for the data transformation and then return the preprocessor object, 
        so that we can use it in the model training part to transform the data before training the model.
        '''
        
        try: # we are using try except block to catch any exception that occurs during the execution of the get_data_transformer_object method, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
            numerical_columns = ['reading_score','writing_score'] # we are creating a list of numerical columns in the data, so that we can apply different transformations to these columns in the pipeline for the data transformation, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
            categorical_columns = ['gender',
                                   'race_ethnicity',
                                   'parental_level_of_education',
                                   'lunch',
                                   'test_preparation_course'
                            ] # we are creating a list of categorical columns in the data, so that we can apply different transformations to these columns in the pipeline for the data transformation, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), # we are adding a step in the pipeline for the numerical columns to fill the missing values in the data by using SimpleImputer with strategy='median', so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
                    ('scaler',StandardScaler()) # we are adding a step in the pipeline for the numerical columns to scale the data by using StandardScaler, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
                ]
            ) # we create a pipeline that does two tasks which is filling the missing values and then scaling the training dataset

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')), # we are adding a step in the pipeline for the categorical columns to fill the missing values in the data by using SimpleImputer with strategy='most_frequent', so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
                    ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')), # we are adding a step in the pipeline for the categorical columns to encode the categories into a one-hot encoded format, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
                    ('scaler',StandardScaler(with_mean=False)) # we are adding a step in the pipeline for the categorical columns to scale the data by using StandardScaler with with_mean=False, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
                ]
            ) # we create a pipeline that does two tasks which is filling the missing values and then encoding the categorical variables
            
            logging.info(f"Numerical columns scaling completed: {numerical_columns}") # we are logging the information message that we have the numerical columns in the data, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            logging.info(f"Categorical columns encoding completed: {categorical_columns}") # we are logging the information message that we have the categorical columns in the data, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.

            #we combine the num and cat pipeline using cat transformer
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns), # we are adding the numerical pipeline to the column transformer and specifying the numerical columns to which we want to apply the transformations in the numerical pipeline, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
                    ('cat_pipeline',cat_pipeline,categorical_columns) # we are adding the categorical pipeline to the column transformer and specifying the categorical columns to which we want to apply the transformations in the categorical pipeline, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
                ]
            ) # we create a column transformer that combines the numerical pipeline and categorical pipeline, so that we can apply different transformations to different columns of the data, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.

            return preprocessor # we are returning the preprocessor object, so that we can use it in the model training part to transform the data before training the model.

        except Exception as e: # if any exception occurs during the execution of the try block, then it will be caught here and assigned to the variable e, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
            logging.info("Error occurred in data transformation component") # we are logging the information message that an error occurred in the data transformation component, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            raise CustomException(e,sys) # we are raising a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file, so that we can get a custom error message with file name and line number where the exception occurred when we print the exception.


    def initiate_data_transformation(self,train_path,test_path): # we are creating a method for the data transformation, which will take the path of the train data and test data as input and then it will create a pipeline for the data transformation and then save the preprocessor object in the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.
        try: # we are using try except block to catch any exception that occurs during the execution of the initiate_data_transformation method, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
            logging.info("Initiated the data transformation component") # we are logging the information message that we have initiated the data transformation component, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            train_df = pd.read_csv(train_path) # we are reading the train data from the path that we have passed as input to the method and assigning it to the train_df variable, so that we can use it to create a pipeline for the data transformation and then save the preprocessor object in the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.
            test_df = pd.read_csv(test_path) # we are reading the test data from the path that we have passed as input to the method and assigning it to the test_df variable, so that we can use it to create a pipeline for the data transformation and then save the preprocessor object in the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.

            logging.info("Read train and test data completed") # we are logging the information message that we have read train and test data, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.

            preprocessor_obj = self.get_data_transformer_object() # we are calling the get_data_transformer_object method to get the preprocessor object, so that we can use it in the model training part to transform the data before training the model.

            logging.info("Obtained preprocessor object") # we are logging the information message that we have obtained preprocessor object, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.

            target_column_name = "math_score"
            numerical_columns = ['reading_score','writing_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1) # we are dropping the target column from the train data and assigning it to the input_feature_train_df variable, so that we can use it to create a pipeline for the data transformation and then save the preprocessor object in the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.
            target_feature_train_df = train_df[target_column_name] # we are selecting the target column from the train data and assigning it to the target_feature_train_df variable, so that we can use it to create a pipeline for the data transformation and then save the preprocessor object in the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.

            # for test dataset
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1) # we are dropping the target column from the test data and assigning it to the input_feature_test_df variable, so that we can use it to create a pipeline for the data transformation and then save the preprocessor object in the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.
            target_feature_test_df = test_df[target_column_name] # we are selecting the target column from the test data and assigning it to the target_feature_test_df variable, so that we can use it to create a pipeline for the data transformation and then save the preprocessor object in the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.

            logging.info("Separated input features and target feature from train and test data") # we are logging the information message that we have separated input features and target feature from train and test data, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) # we are fitting the preprocessor object on the input features of the train data and then transforming the input features of the train data, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) # we are transforming the input features of the test data using the preprocessor object that we have fitted on the input features of the train data, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.
            logging.info("Transformed input features of train and test data") # we are logging the information message that we have transformed input features of train and test data, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # we are concatenating the transformed input features of the train data and the target feature of the train data to create a new array for the train data, so that we can use it in the model training part to train the model.
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # we are concatenating the transformed input features of the test data and the target feature of the test data to create a new array for the test data, so that we can use it in the model training part to train the model.
            # here .c_ is used to concatenate the input features and target feature of the train and test data, so that we can keep our data in a good shape and also we can keep our data in a good format for the model training part, so that we can get good results from the model training part.

            logging.info("Concatenated input features and target feature of train and test data") # we are logging the information message that we have concatenated input features and target feature of train and test data, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            logging.info("Saved preprocessor object") # we are logging the information message that we have saved preprocessor object, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.

            #saving pickle file of preprocessor object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, # we are passing the file path where we want to save the preprocessor object, which is the path that we have created in the data transformation configuration, so that we can use it in the model training part to transform the data before training the model.
                obj = preprocessor_obj # we are passing the preprocessor object that we want to save in the pickle file, so that we can use it in the model training part to transform the data before training the model.
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path# this is pickle file path where we will save the preprocessor object, so that we can use it in the model training part to transform the data before training the model.
            )
        except Exception as e: # if any exception occurs during the execution of the try block, then it will be caught here and assigned to the variable e, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
            logging.info("Error occurred in initiate_data_transformation") # we are logging the information message that an error occurred in the initiate_data_transformation method, so that we can keep track of the execution of our project in the log file that we have configured in the logger.py file.
            raise CustomException(e,sys) # we are raising a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file, so that we can get a custom error message with file name and line number where the exception occurred when we print the exception. 


