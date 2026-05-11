## here we will train using different models and see which model gives us the best result ie r2 score
# and in every such problems we will be using this method of trying out different models and see which one 
# gives us the best result 
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

# now for every code we need to creare a config file and in that config file we will be saving the path where we want to save our model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            #evaluate_model is a func we create in the utils.py file and in that func we will be evaluating all the models and then we will be getting the best model score and best model name
            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # to get the best model score from the model_report dictionary
            best_model_score = max(sorted(model_report.values()))
            # to get the best model name from the model_report dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # we put a threshold of 0.6 for the best model score, if the best model score is less than 0.6 then we will raise an exception that no best model found with score greater than 0.6, so that we can avoid any error that might occur due to using a model with low score, and also we can get a better result by using a model with high score.
            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with score greater than 0.6"
                )
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )  # we are saving the best model in the artifacts folder, so that we can use it in the model evaluation part, and also we can use it in the future to make predictions on new data.          
            predicted = best_model.predict(X_test) # we are predicting the target variable for the testing data, so that we can evaluate the model on the testing data.

            r2_square = r2_score(y_test, predicted) # we are calculating the r2 score for the testing data, so that we can evaluate the model on the testing data.
            return r2_square # we are returning the r2 score for the testing data, so that we can evaluate the model on the testing data.
        
        
        
        except Exception as e:
            raise CustomException(e, sys)
