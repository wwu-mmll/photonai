import logging
import sys

import sklearn

logging.getLogger(sklearn.__name__).setLevel(logging.ERROR)

dask_logger = logging.getLogger('distributed.utils_perf')
dask_logger.setLevel(logging.ERROR)
for handler in dask_logger.handlers:
    handler.setLevel(logging.ERROR)

# create photon logger
logger = logging.getLogger('PHOTON')
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


# add custom log level
def verbose(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    self._log(VERBOSE_LEVELV_NUM, message, args, **kws)


VERBOSE_LEVELV_NUM = 15
logging.addLevelName(VERBOSE_LEVELV_NUM, "VERBOSE")
logging.Logger.verbose = verbose



