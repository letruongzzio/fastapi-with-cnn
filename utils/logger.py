import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from config.logging_cfg import LoggingConfig

class Logger:
    """
    A utility class for logging messages to both console and file.

    Steps:
        1. Initialize the Logger instance with a name, log level, and optional log file.
        2. Call get_logger() to set up the logger with the specified log level and handlers.
        3. Use log_model() to log the name of the predictor.
        4. Use log_response() to log prediction probability, ID, and class.

    Attributes:
        log (logging.Logger): The logger instance.
    
    Methods:
        log_model(predictor_name): Logs the name of the predictor.
        log_response(pred_prob, pred_id, pred_class): Logs prediction details.
    
    Example:
        >>> logger = Logger(name="MyLogger", log_level=logging.DEBUG, log_file="app.log")
        >>> logger.log_model("ResNet-50")
        >>> logger.log_response(0.95, 1, "Dog")
    """
    
    def __init__(self, name="", log_level=logging.INFO, log_file=None) -> None:
        """Initializes the Logger instance."""
        self.log = logging.getLogger(name)
        self.log.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        if log_file is not None:
            # Using `RotatingFileHandler` to rotate log files when they reach a certain size.
            file_handler = RotatingFileHandler(
                os.path.join(LoggingConfig.LOG_DIR, log_file),
                maxBytes=10000, # `maxBytes` is the maximum size of the log file before it is rotated.
                backupCount=10 # `backupCount` is the number of backup log files to keep.
            )
            file_handler.setFormatter(formatter) # Set the formatter for the file handler.
            self.log.addHandler(file_handler) # Add the file handler to the logger.
        else:
            # Using `StreamHandler` to log messages to the console.
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self.log.addHandler(stream_handler)
    
    def log_model(self, predictor_name):
        """Logs the name of the predictor being used."""
        self.log.info("Predictor name: %s", predictor_name)
    
    def log_response(self, pred_prob, pred_id, pred_class):
        """Logs prediction probability, ID, and class."""
        self.log.info("Predicted Prob: %s - Predicted ID: %s - Predicted Class: %s", pred_prob, pred_id, pred_class)
