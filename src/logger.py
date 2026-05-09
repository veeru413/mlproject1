# use of this is to log all the executions and nformation in some files, so that we can keep track of all the 
# executions and information in our project, and also we can use it for debugging purposes, 
# so that we can easily find out where the error occurred and what was the error message, 
# and also we can use it for monitoring purposes, so that we can monitor the performance of our 
# project and also we can use it for auditing purposes, so that we can keep track of all the 
# changes made to our project and also we can use it for troubleshooting purposes, 
# so that we can easily find out what went wrong in our project and how to fix it.
import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log" # we are creating a log file with the current date and time as the name of the file, so that we can keep track of all the logs in different files based on the date and time of execution.
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE) # we are creating a path for the log file by joining the current working directory, logs folder and the log file name.
os.makedirs(os.path.dirname(logs_path),exist_ok=True) # we are creating the logs folder if it does not exist, and also we are using exist_ok=True to avoid any error if the folder already exists.

LOG_FILE_PATH = logs_path # we are creating a path for the log file by joining the logs path and the log file name, so that we can use this path to configure the logging module.

logging.basicConfig(
    filename=LOG_FILE_PATH, # we are configuring the logging module to write the logs to the log file path that we have created.
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", # we are configuring the logging module to write the logs in a specific format, which includes the date and time of the log, the line number where the log was generated, the name of the logger, the level of the log and the log message.
    level=logging.INFO # we are configuring the logging module to write the logs with the level of INFO and above, which means that it will write the logs with the level of INFO, WARNING, ERROR and CRITICAL.
)

