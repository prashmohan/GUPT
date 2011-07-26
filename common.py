import time
import logging

logger = logging.getLogger(__name__)

def profile_func(func):
    def measured_func(*args, **kwargs):
        ctr_start = time.time()
        ret_val = func(*args, **kwargs)
        ctr_stop = time.time()
        logger.debug("PROFILE: " + func.__name__ + " " + str((ctr_stop - ctr_start) * 1000))
        return ret_val
    return measured_func

def isiterable(record):
    return getattr(record, '__iter__', False)

