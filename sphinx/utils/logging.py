import logging
from logging import Logger
import sys

def get_logger(name:str, level) -> Logger:
    log_fmt = '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
    formatter = logging.Formatter(log_fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=level,
                        format=log_fmt,
                        datefmt='%m-%d %H:%M:%S',
                        handlers=[console_handler])
    log = logging.getLogger(name)
    return log