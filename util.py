import logging
import re

BIG_DIVIDER = "=================\n"
SMALL_DIVIDER = "-----------------\n"

MODEL_FOLDER_PATH = "models/"


def setup_logger_handlers(log_filename, log_mode='a', c_display=False, c_logger_level=logging.INFO):
    f_handler = logging.FileHandler(log_filename, mode=log_mode)
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # f_format = logging.Formatter('%(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    # default handler level are info for terminal output
    # and debug for the log output
    c_handler.setLevel(c_logger_level)
    f_handler.setLevel(logging.DEBUG)

    # if the logger exist, it does not create a new one
    handlers = [f_handler]
    if c_display:
        handlers.append(c_handler)
    return handlers


def setup_logger(name, handlers=[], logger_level=logging.INFO):
    """To setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logger.handlers = handlers
    logger.setLevel(logger_level)

    return logger


def regex_search(regex, string, logger=None):
    result = re.findall(regex, string, re.M)
    if logger and not result :
        logger.error(f"result not found: \"{regex}\"")
        exit(0)
    return result


