import datetime
import logging
import sys
import sklearn

logging.getLogger(sklearn.__name__).setLevel(logging.ERROR)

dask_logger = logging.getLogger('distributed.utils_perf')
dask_logger.setLevel(logging.ERROR)
for handler in dask_logger.handlers:
    handler.setLevel(logging.ERROR)

# create photon logger
logger = logging.getLogger('PHOTONAI')
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)
logger.propagate = False

VERBOSE_LEVELV_NUM = 25
CLEAN_LEVELV_NUM = 21
INFO_LEVELV_NUM = 20
DEBUG_LEVELV_NUM = 10
LENGTH = 101


# add custom log level
def photon_system_log(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE_LEVELV_NUM):
        self._log(VERBOSE_LEVELV_NUM, message, args, **kws)


def star_log(self):
    photon_system_log(self, '*' * LENGTH)


def system_line(self):
    photon_system_log(self, '-' * LENGTH)


def underscore_log(self):
    clean_info(self, '-' * LENGTH)


def clean_info(self, message, *args, **kws):
    if self.isEnabledFor(CLEAN_LEVELV_NUM):
        self._log(CLEAN_LEVELV_NUM, message, args, **kws)


def info(self, message, *args, **kws):
    if self.isEnabledFor(INFO_LEVELV_NUM):
        timestamp = datetime.datetime.now()
        log_message = timestamp.strftime("%d/%m/%Y-%H:%M:%S")
        if message:
            log_message += " | " + str(message)
        self._log(INFO_LEVELV_NUM, log_message, args, **kws)


def debug(self, message, *args, **kws):
    if self.isEnabledFor(DEBUG_LEVELV_NUM):
        timestamp = datetime.datetime.now()
        if isinstance(message, str):
            log_message = timestamp.strftime("%d/%m/%Y-%H:%M:%S")
            log_message += " | " + str(message)
        else:
            log_message = message
        self._log(DEBUG_LEVELV_NUM, log_message, args, **kws)


logging.addLevelName(VERBOSE_LEVELV_NUM, "PHOTON_SYSTEM_LOG")
logging.Logger.photon_system_log = photon_system_log
logging.addLevelName(CLEAN_LEVELV_NUM, "CLEAN_INFO")
logging.Logger.clean_info = clean_info
logging.addLevelName(INFO_LEVELV_NUM, "INFO")
logging.Logger.info = info
logging.addLevelName(DEBUG_LEVELV_NUM, "DEBUG")
logging.Logger.debug = debug
logging.Logger.stars = star_log
logging.Logger.line = underscore_log
logging.Logger.system_line = system_line
