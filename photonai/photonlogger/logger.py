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

VERBOSE_LEVELV_NUM = 25


# add custom log level
def photon_system_log(self, message, *args, **kws):
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(VERBOSE_LEVELV_NUM):
        self._log(VERBOSE_LEVELV_NUM, message, args, **kws)


logging.addLevelName(VERBOSE_LEVELV_NUM, "PHOTON_SYSTEM_LOG")
logging.Logger.photon_system_log = photon_system_log



