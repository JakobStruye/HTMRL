import logging

formatter = logging.Formatter(fmt='%(message)s')

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

logging.addLevelName(5, "TRACE")
logger.setLevel(20) #default

def set_debug():
    logger.setLevel("DEBUG")

def set_trace():
    logger.setLevel("TRACE")

def has_debug():
    return logger.getEffectiveLevel() <= 10

def has_trace():
    return logger.getEffectiveLevel() <= 5

def debug(msg, *args):

    if has_debug() and args:
        for arg in args:
            msg += str(arg)
    logger.debug(msg)

def trace(msg, *args):

    if has_trace() and args:
        for arg in args:
            msg += str(arg)
    logger.log(5, msg)