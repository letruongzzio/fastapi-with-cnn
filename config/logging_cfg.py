import os

ROOT_DIR = os.path.expanduser('~/fastapi-with-cnn')

class LoggingConfig:
    """
    Configuration class for logging
    """
    LOG_DIR = os.path.join(ROOT_DIR, "logs")

os.makedirs(LoggingConfig.LOG_DIR, exist_ok=True)
