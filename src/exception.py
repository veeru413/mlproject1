###this code for exception is used wherever we use the try catch block in our project, 
# and inside the catch block
# we will raise the CustomException and pass the error and error_detail to it, so that we can get the custom 
# error message with file name and line number where the exception occurred.

import logging
import sys #sys module is used to get the exception information

def error_message_detail(error,error_detail:sys): # here :sys means that the error_detail parameter is of type sys
    _,_,exc_tb = error_detail.exc_info() # exc_info() function returns a tuple containing the exception type, exception value and traceback object. We are only interested in the traceback object here, so we unpack the tuple and assign it to exc_tb variable.
    file_name = exc_tb.tb_frame.f_code.co_filename # we can get the file name where the exception occurred from the traceback object using tb_frame attribute and then f_code attribute and then co_filename attribute.
    line_number = exc_tb.tb_lineno # we can get the line number where the exception occurred from the traceback object using tb_lineno attribute.
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with error message: {str(error)}" # we can create a custom error message using f-string to include the file name, line number and error message.
    return error_message

class CustomException(Exception): # we are creating a custom exception class that inherits from the built-in Exception class.
    def __init__(self,error,error_detail:sys): # we are overriding the __init__ method of the Exception class to accept two parameters, error and error_detail.
        super().__init__(error) # we are calling the __init__ method of the parent class (Exception) to initialize the error message.
        self.error_message = error_message_detail(error,error_detail=error_detail) # we are calling the error_message_detail function to get the custom error message and assigning it to the error_message attribute of the CustomException class.

    def __str__(self): # we are overriding the __str__ method of the Exception class to return the custom error message when the exception is printed.
        return self.error_message
    
