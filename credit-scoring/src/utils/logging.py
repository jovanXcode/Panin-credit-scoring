import logging
import logging.config 
from pydantic import BaseModel

class LogConfig(BaseModel):
    """Logging configuration to be set for the server"""

    LOGGER_NAME: str = "mycoolapp"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "INFO"

    # Logging config
    version: int = 1
    disable_existing_loggers: bool = True
    formatters: dict = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "default": {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },

    }
    loggers: dict = {
        LOGGER_NAME: {"handlers": ["default"], "level": LOG_LEVEL},
    }

def get_logger(name: str):
   
    log_config = LogConfig(LOGGER_NAME=name)
    logging.config.dictConfig(log_config.dict())

    logger = logging.getLogger(name)

    return logger

# import logging.config

# LOGGING = {
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "json": {
#             "format": "%(asctime)s %(levelname)s %(message)s",
#             "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
#         }
#     },
#     "handlers": {
#         "stdout": {
#             "class": "logging.StreamHandler",
#             "stream": "ext://sys.stdout",
#             "formatter": "json",
#         }
#     },
#     "loggers": {"": {"handlers": ["stdout"], "level": "DEBUG"}},
# }


# logging.config.dictConfig(LOGGING)
