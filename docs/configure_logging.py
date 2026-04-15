import logging
import os.path
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug, wrap_embedding_func_with_attrs

def configure_logging():
    for name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        inst = logging.getLogger(name)
        inst.handlers = []
        inst.filters = []
    log_file = os.path.join(os.getcwd(), "lightrag_query.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(levelname)s: %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "file":{
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file,
                "maxBytes": 10485760,
                "backupCount": 5,
                "encoding": "utf-8",
            }
        },
        "loggers": {
            "lightrag": {
                "handlers": ["file", "console"],
                "level": "INFO",
                "propagate": False,
            }
        }
    })
    logger.setLevel(logging.DEBUG)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "False").lower() == "true")
