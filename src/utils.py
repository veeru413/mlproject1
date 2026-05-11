## here the use of utils.py file is to keep all the utility functions that we will use in our project, 
# so that we can keep our code clean and organized, and also we can reuse the utility functions in 
# different parts of our project without having to write the same code again and again, so that we can 
# save time and also we can avoid any error that might occur due to writing the same code again and again.

# utils will have all the common functionalities that we can use in the eniteirty of the project

import os
import sys
import pandas as pd
import numpy as np
import dill # its a library that is used to save the model and the preprocessor objects in the artifacts folder, so that we can use it in the model training and model evaluation part.

from src.exception import CustomException # we are importing the CustomException class from the exception.py file that we have created in the src folder, so that we can use it to raise custom exceptions in our project.
from sklearn.metrics import r2_score # we are importing the r2_score function from the sklearn.metrics module, so that we can use it to evaluate the models in the model training part, so that we can get the best model score and best model name.
from sklearn.model_selection import GridSearchCV # we are importing the GridSearchCV class from the sklearn.model_selection module, so that we can use it to perform hyperparameter tuning for the models in the model training part, so that we can get the best hyperparameters for the models, so that we can get the best model score and best model name.

def save_object(file_path,obj): # we are creating a function to save the object in a file, which will be used to save the model and the preprocessor objects in the artifacts folder, so that we can use it in the model training and model evaluation part.
    try:
        dir_path = os.path.dirname(file_path) # we are getting the directory path from the file path, so that we can create the directory if it does not exist, and also we can use it to save the object in the file.
        os.makedirs(dir_path,exist_ok=True) # we are creating the directory if it does not exist, and also we are using exist_ok=True to avoid any error if the directory already exists, so that we can save the object in the file without any error.
        
        with open(file_path,'wb') as file_obj: # we are opening the file in write binary mode, so that we can save the object in the file.
            dill.dump(obj,file_obj) # we are using pickle module to save the object in the file, so that we can use it in the model training and model evaluation part.


    except Exception as e: # if any exception occurs during the execution of the try block, then it will be caught here and assigned to the variable e, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
        raise CustomException(e,sys) # we are raising a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file, so that we can get a custom error message with file name and line number where the exception occurred when we print the exception.
    

def evaluate_model(X_train,y_train,X_test,y_test,models,params): # we are creating a function to evaluate the models, which will be used to evaluate the models in the model training part, so that we can get the best model score and best model name.
    try:
        report = {} # we are creating an empty dictionary to store the model name and model score, so that we can get the best model score and best model name.

        for i in range(len(models)): # we are iterating through the models dictionary, so that we can evaluate each model on the testing data, and also we can get the model name and model score for each model.
            model = list(models.values())[i] # we are getting the model object from the models dictionary, so that we can fit the model on the training data and evaluate the model on the testing data.
            para = params[list(models.keys())[i]] # we are getting the hyperparameters for the model from the params dictionary, so that we can use it to perform hyperparameter tuning for the model, and also we can get the best hyperparameters for the model, so that we can get the best model score and best model name.

            gs = GridSearchCV(model,para,cv=3) # we are creating an object of the GridSearchCV class, and passing the model object, hyperparameters and cv=3 to it, so that we can perform hyperparameter tuning for the model using grid search cross validation, and also we can get the best hyperparameters for the model, so that we can get the best model score and best model name.
            gs.fit(X_train,y_train) # we are fitting the GridSearchCV object on the training data, so that we can perform hyperparameter tuning for the model using grid search cross validation, and also we can get the best hyperparameters for the model, so that we can get the best model score and best model name.

            model.set_params(**gs.best_params_) # we are setting the best hyperparameters for the model, so that we can get the best model score and best model name.
            model.fit(X_train,y_train) # we are fitting the model on the training data, so that we can evaluate the model on the testing data.

            #model.fit(X_train,y_train) # we are fitting the model on the training data, so that we can evaluate the model on the testing data.
            y_test_pred = gs.predict(X_test) # we are predicting the target variable for the testing data, so that we can evaluate the model on the testing data.
            train_model_score = r2_score(y_train,gs.predict(X_train)) # we are calculating the r2 score for the training data, so that we can evaluate the model on the training data, and also we can compare the training score and testing score to check for overfitting or underfitting.
            test_model_score = r2_score(y_test,y_test_pred) # we are calculating the r2 score for the testing data, so that we can evaluate the model on the testing data.
            report[list(models.keys())[i]] = test_model_score # we are storing the model name and model score in the report dictionary, so that we can get the best model score and best model name.

        return report # we are returning the report dictionary, so that we can get the best model score and best model name in the model training part.
    except Exception as e: # if any exception occurs during the execution of the try block, then it will be caught here and assigned to the variable e, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
        raise CustomException(e,sys) # we are raising a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file, so that we can get a custom error message with file name and line number where the exception occurred when we print the exception.

def load_object(file_path): # we are creating a function to load the object from a file, which will be used to load the model and the preprocessor objects from the artifacts folder, so that we can use it in the model training and model evaluation part.
    try:
        with open(file_path,'rb') as file_obj: # we are opening the file in read binary mode, so that we can load the object from the file.
            return dill.load(file_obj) # we are using pickle module to load the object from the file, so that we can use it in the model training and model evaluation part.

    except Exception as e: # if any exception occurs during the execution of the try block, then it will be caught here and assigned to the variable e, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
        raise CustomException(e,sys) # we are raising a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file, so that we can get a custom error message with file name and line number where the exception occurred when we print the exception.


