import datetime
import inspect
from enum import Enum
from functools import total_ordering

from photonai.helper.helper import Singleton


@Singleton
class Logger:
    def __init__(self):
        # load configuration
        self.config = {
            'print_to_console': True,
            'print_to_file': True
        }

        # handle multiple instances of hyperpipe
        self.loggers = []

        # Set default LogLevel
        # Should be LogLevel.WARN!
        # Is LogLevel.DEBUG now only for testing purposes
        self._log_level = self.LogLevel.INFO
        self.verbosity_level = 0

        self.log_level_console = self.LogLevel.INFO
        self.log_level_file = self.LogLevel.INFO

        # Should the log also be printed to the console?
        # Recommendation: Set to True during development, false in production-environment
        self._print_to_console = self.config['print_to_console']

    @staticmethod
    def set_print_to_console(self, status: bool):
        self._print_to_console = status

    def set_log_level(self, level):
        """" Use this method to change the log level. """
        self._log_level = level

    def set_custom_log_file(self, logfile):
        self._logfile_name = logfile

    def set_verbosity(self, verbose=0):
        """ Use this method to change the log level from verbosity attribute of hyperpipe. """
        self.verbosity_level = verbose
        if verbose == 0:
            self.set_log_level(self.LogLevel.INFO)
        elif verbose == 1:
            self.set_log_level(self.LogLevel.VERBOSE)
        elif verbose == 2:
            self.set_log_level(self.LogLevel.DEBUG)
        else:
            self.set_log_level(self.LogLevel.WARN)

    # Debug should be used for information that may be useful for program-debugging (most information
    # i.e. training epochs of neural nets
    def debug(self, message: str):
        if self._log_level <= self.LogLevel.DEBUG:
            self._distribute_log(message, 'DEBUG')

    # Verbose should be used if something interesting (but uncritically) happened
    # i.e. every hp config that is tested
    def verbose(self, message: str):
        if self._log_level <= self.LogLevel.VERBOSE:
            self._distribute_log(message, 'VERBOSE')

    # Info should be used if something interesting (but uncritically) happened
    # i.e. most basic info on photonai hyperpipe
    def info(self, message: str):
        if self._log_level <= self.LogLevel.INFO:
            self._distribute_log(message, 'INFO')

    # Something may have gone wrong? Use warning
    def warn(self, message: str):
        if self._log_level <= self.LogLevel.WARN:
            self._distribute_log(message, 'WARN')

    # Something broke down. Error should be used if something unexpected happened
    def error(self, message: str):
        if self._log_level <= self.LogLevel.ERROR:
            self._distribute_log(message, 'ERROR')

    # Takes a message and inserts it into the given collection
    # Handles possible console-photonlogger
    def _distribute_log(self, message: str, log_type: str):

        entry = self._generate_log_entry(message, log_type)

        if self._print_to_console:
            self._print_entry(entry)

    @staticmethod
    def _print_entry(entry: dict):
        date_str = entry['logged_date'].strftime("%Y-%m-%d %H:%M:%S")
        #print("{0} UTC - {1}: {2}".format(date_str, entry['log_type'], entry['message']))
        print("{0}".format(entry['message']))

    @staticmethod
    def _generate_log_entry(message: str, log_type: str):
        """Todo: Get current user from user-service and add username to log_entry"""
        log_entry = {'log_type': log_type,
                     'logged_date': datetime.datetime.utcnow(),
                     'message': message}
        if inspect.stack()[3][3]:
            # If the call stack is changed the first array selector has to be changed
            log_entry['called_by'] = inspect.stack()[3][3]
        else:
            log_entry['called_by'] = 'Unknown caller'

        return log_entry

    def store_logger_names(self, name):
        return self.loggers.append(name)

    # Definition of LogLevels, the lower the number, the stronger the photonlogger will be
    @total_ordering
    class LogLevel(Enum):
        ERROR = 4
        WARN = 3
        INFO = 2
        VERBOSE = 1
        DEBUG = 0

        def __lt__(self, other):
            if self.__class__ is other.__class__:
                return self.value < other.value
            return NotImplemented


if __name__ == "__main__":
    logger = Logger()
    logger.set_verbosity(2)

    logger.debug('test-Log debug message')
    logger.info('test-Log info message')
    logger.warn('test-Log warn message')
    logger.error('test-Log error message')

    # Starting here, only warn and error logs
    # should be created
    logger.set_verbosity(0)
    logger.debug('test-Log debug message')
    logger.info('test-Log info message')
    logger.warn('test-Log warn message')
    logger.error('test-Log error message')
