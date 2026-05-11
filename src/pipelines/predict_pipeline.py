# this is the most important file in our project, because this is the file where we will be calling all the components that we have created in our project, and we will be calling the main function of each component in this file, so that we can run our entire project from this file, and also we can test our entire project from this file, so that we can get the final output of our project, which is the trained model for our model evaluation part.
# here we would create a web application which would have a form where we would take the input from the user and 
# then we would use the trained model to predict the output for the input given by the user, 
# so that we can get the final output of our project, which is the predicted value for the input given by the user, 
# so that we can make our project more interactive and user friendly, so that we can get more insights from our project 
# and also we can make our project more useful for the users.
# btw the web is on the app.py file
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features): # this function will be responsible to take the input data from the user, and then we will use the trained model to predict the output for the input data given by the user, and then we will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
        try:
            model_path = 'artifacts/model.pkl' # this is the path where we have saved our trained model, which we will use to predict the output for the input data given by the user, and then we will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
            preprocessor_path = 'artifacts/preprocessor.pkl' # this is the path where we have saved our preprocessor object, which we will use to preprocess the input data given by the user, so that we can get better performance from our model, and then we will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
            # preprocessor.pkl is the file where we have saved our preprocessor object, which we have created in our data transformation part, and we will use this preprocessor object to preprocess the input data given by the user, so that we can get better performance from our model, and then we will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
            model = load_object(file_path=model_path) # this function will be responsible to load the trained model from the path where we have saved it, and then we will use this model to predict the output for the input data given by the user, and then we will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
            preprocessor = load_object(file_path=preprocessor_path) # this function will be responsible to load the preprocessor object from the path where we have saved it, and then we will use this preprocessor object to preprocess the input data given by the user, so that we can get better performance from our model, and then we will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
            data_scaled = preprocessor.transform(features) # this function will be responsible to preprocess the input data given by the user, so that we can get better performance from our model, and then we will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
            pred = model.predict(data_scaled) # this function will be responsible to predict the output for the input data given by the user, and then we will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
            return pred # this will return the predicted output to the user, so that we can display the predicted output to the user in our web application.
        
        except Exception as e: # if any exception occurs during the execution of the try block, then it will be caught here and assigned to the variable e, so that we can raise a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file.
            raise CustomException(e,sys) # we are raising a custom exception with the error message and error details using the CustomException class that we have created in the exception.py file, so that we can get a custom error message with file name and line number where the exception occurred when we print the exception.

class CustomData: # will be responsible to map the input data from the user to the format that our model can understand, and also we will use this class to create a dataframe from the input data given by the user, so that we can pass this dataframe to our model for prediction.
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_dataframe(self): # this function will be responsible to create a dataframe from the input data given by the user, so that we can pass this dataframe to our model for prediction.
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

