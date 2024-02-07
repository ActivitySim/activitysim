from __future__ import annotations

import logging


def error_logging(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logging.error(f"===== ERROR IN {func.__name__} =====")
            logging.exception(f"{err}")
            logging.error("===== / =====")
            raise

    return wrapper
