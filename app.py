from flask import Flask, request, render_template # we are importing the Flask class from the flask module, which is a micro web framework for python, and we are also importing the request and render_template functions from the flask module, which we will use to handle the HTTP requests and render the HTML templates for our web application.
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler # we are importing the StandardScaler class from the sklearn.preprocessing module, which we will use to scale the input data for our model, so that we can get better performance from our model.
from src.pipelines.predict_pipeline import CustomData, PredictPipeline # we are importing the CustomData class and PredictPipeline class from the predict_pipeline.py file that we have created in the src/pipelines folder, which we will use to handle the prediction pipeline for our web application, so that we can get the predicted output for the input data given by the user, and then we can display the predicted output to the user in our web application.


application = Flask(__name__) # we are creating an object of the Flask class, and passing the name of the module to it, so that we can use it to run our web application.
app = application # we are creating an alias for the application object, so that we can use it to run our web application.


# route for the home page

@app.route('/') # we are defining a route for the home page of our web application, which will be accessed when the user visits the root URL of our web application, and we will render the home.html template for this route, which will contain a form to take input from the user for prediction.
def index():
    return render_template('index.html') # this searches for template folder and then index.html file in that template folder and renders it, so that we can display the home page of our web application to the user, which will contain a form to take input from the user for prediction.

@app.route('/predict', methods=['GET','POST']) # we are defining a route for the predict page of our web application, which will be accessed when the user submits the form on the home page of our web application, and we will handle the POST request for this route, which will contain the input data from the user for prediction, and we will use the trained model to predict the output for the input data given by the user, and then we will render the result.html template to display the predicted output to the user.
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html') # if the request method is GET, then we will render the home.html template again, so that the user can submit the form again for prediction.
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )
        pred_df = data.get_data_as_dataframe() # this function will create a dataframe from the input data given by the user, so that we can pass this dataframe to our model for prediction.
        print(pred_df) # this will print the dataframe created from the input data given by the user, so that we can check if the dataframe is created correctly or not.
        predict_pipeline = PredictPipeline() # we are creating an object of the PredictPipeline class, which we have created in the predict_pipeline.py file, so that we can use this object to handle the prediction pipeline for our web application, so that we can get the predicted output for the input data given by the user, and then we can display the predicted output to the user in our web application.
        pred = predict_pipeline.predict(pred_df) # this function will take the dataframe created from the input data given by the user, and then it will use the trained model to predict the output for the input data given by the user, and then it will return the predicted output to us, so that we can display the predicted output to the user in our web application.
        return render_template('home.html', results=pred[0]) # this will render the result.html template, and we will pass the predicted output to the template using the prediction variable, so that we can display the predicted output to the user in our web application.
    
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True) # we are running the web application in debug mode, so that we can get the error messages in the console if any error occurs during the execution of the web application, and also we can get the changes reflected in the web application without having to restart the web application, so that we can develop our web application faster and more efficiently.







