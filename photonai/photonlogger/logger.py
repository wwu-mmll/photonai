import logging
import sys
import datetime
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
CLEAN_LEVELV_NUM = 21
INFO_LEVELV_NUM = 20
DEBUG_LEVELV_NUM = 10


# add custom log level
def photon_system_log(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE_LEVELV_NUM):
        self._log(VERBOSE_LEVELV_NUM, message, args, **kws)


def clean_info(self, message, *args, **kws):
    if self.isEnabledFor(CLEAN_LEVELV_NUM):
        self._log(CLEAN_LEVELV_NUM, message, args, **kws)


def info(self, message, *args, **kws):
    if self.isEnabledFor(INFO_LEVELV_NUM):
        timestamp = datetime.datetime.now()
        log_message = timestamp.strftime("%d/%m/%Y-%H:%M:%S")
        if message:
            log_message += " | " + message
        self._log(INFO_LEVELV_NUM, log_message, args, **kws)


def debug(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_LEVELV_NUM):
        timestamp = datetime.datetime.now()
        log_message = timestamp.strftime("%d/%m/%Y-%H:%M:%S")
        if message:
            log_message += " | " + message
        self._log(DEBUG_LEVELV_NUM, log_message, args, **kws)


logging.addLevelName(VERBOSE_LEVELV_NUM, "PHOTON_SYSTEM_LOG")
logging.Logger.photon_system_log = photon_system_log
logging.addLevelName(CLEAN_LEVELV_NUM, "CLEAN_INFO")
logging.Logger.clean_info = clean_info
logging.addLevelName(INFO_LEVELV_NUM, "INFO")
logging.Logger.info = info
logging.addLevelName(DEBUG_LEVELV_NUM, "DEBUG")
logging.Logger.debug = debug


