import inspect
from enum import Enum
from functools import total_ordering

from pymongo import MongoClient
from pymongo import ASCENDING
import datetime

""" Logging is a simple way to emit and store logs. 

    The default LogLevel is WARN. It should only be increased 
    (to info or debug) if you need more detailed information,
    because extensive logging significantly impacts performance
    and clutters the database. The logs can also be printed on
    the console by setting print_to_console to True.
    
    Usage: 
    1) Import with
        from Logging import Logger
    2) Log with
        Logger.debug('Logging message!')
"""


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    from https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def __call__(self, hyperpipe_name=None):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            if hyperpipe_name:
                self._instance.store_logger_names(name=hyperpipe_name)
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            if hyperpipe_name:
                self._instance.store_logger_names(name=hyperpipe_name)
            return self._instance

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class Logger:

    def __init__(self):
        try:
            # Create the photon_log_db
            self.client = MongoClient('localhost', 27017)
            self.log_db = self.client.your_collection

            # All collections should probably be capped so that no manual log-pruning is necessary
            # like info_log_collection = log_db.createCollection("info_log", capped=True, size=15000)?!
            # info_log_collection = log_db.createCollection("info_log", capped=True, size=15000)
            self.debug_log_collection = self.log_db.debug_log
            self.debug_log_collection.ensure_index([('logged_date', ASCENDING)])

            self.verbose_log_collection = self.log_db.verbose_log
            self.verbose_log_collection.ensure_index([('logged_date', ASCENDING)])

            self.info_log_collection = self.log_db.info_log
            self.info_log_collection.ensure_index([('logged_date', ASCENDING)])

            self.warn_log_collection = self.log_db.warn_log
            self.warn_log_collection.ensure_index([('logged_date', ASCENDING)])

            self.error_log_collection = self.log_db.error_log
            self.error_log_collection.ensure_index([('logged_date', ASCENDING)])

            self.use_db = True
        except NotImplementedError as e:
            print('WARNING: Error setting up MongoDB. Only using .txt logging.')
            self.use_db = False

        # handle multiple instances of hyperpipe
        self.loggers = []


        # Set default LogLevel
        # Should be LogLevel.WARN!
        # Is LogLevel.DEBUG now only for testing purposes
        self._log_level = self.LogLevel.INFO

        # Should the log also be printed to the console?
        # Recommendation: Set to True during development, false in production-environment
        self._print_to_txt = True
        self._logfile_name = 'photon.log'
        with open(self._logfile_name, "w") as text_file:
            text_file.write('PHOTON LOGFILE - ' + str(datetime.datetime.utcnow()))

    def set_print_to_console(self, status: bool):
        self._print_to_console = status


    def set_log_level(self, level):
        """" Use this method to change the log level. """
        self._log_level = level

    def set_verbosity(self, verbose=0):
        """ Use this method to change the log level from verbosity attribute of hyperpipe. """
        if verbose == 0:
            self.set_log_level(self.LogLevel.INFO)
        elif verbose == 1:
            self.set_log_level(self.LogLevel.VERBOSE)
        elif verbose == 2:
            self.set_log_level(self.LogLevel.DEBUG)
        else:
            self.set_log_level(self.LogLevel.WARN)


        # Debug should be used for information that may be useful for program-debugging (most information)

    # i.e. training epochs of neural nets
    def debug(self, message: str):
        if (self._log_level <= self.LogLevel.DEBUG):
            self._insert_log_into_database(message, self.debug_log_collection,
                                      'DEBUG')

    # Verbose should be used if something interesting (but uncritically) happened
    # i.e. every hp config that is tested
    def verbose(self, message: str):
        if (self._log_level <= self.LogLevel.VERBOSE):
            self._insert_log_into_database(message, self.verbose_log_collection,
                                      'VERBOSE')

    # Info should be used if something interesting (but uncritically) happened
    # i.e. most basic info on photon hyperpipe
    def info(self, message: str):
        if (self._log_level <= self.LogLevel.INFO):
            self._insert_log_into_database(message, self.info_log_collection,
                                      'INFO')

    # Something may have gone wrong? Use warning
    def warn(self, message: str):
        if (self._log_level <= self.LogLevel.WARN):
            self._insert_log_into_database(message, self.warn_log_collection,
                                      'WARN')

    # Something broke down. Error should be used if something unexpected happened
    def error(self, message: str):
        if (self._log_level <= self.LogLevel.ERROR):
            self._insert_log_into_database(message, self.error_log_collection,
                                      'ERROR')

    # Takes a message and inserts it into the given collection
    # Handles possible console-logging
    def _insert_log_into_database(self, message: str, collection,
                                  log_type: str):

        entry = self._generate_log_entry(message, log_type)
        self._print_entry(entry)
        if self._print_to_txt:
            with open(self._logfile_name, "a", newline='\n') as text_file:
                text_file.write('\n')
                text_file.write(str(entry['message']))
        if (self._log_level > self.LogLevel.INFO) and self.use_db:
            collection.insert(entry)

    def _print_entry(self, entry: dict):
        print(entry['message'])

    def _generate_log_entry(self, message: str, log_type: str):
        """Todo: Get current user from user-service and add username to log_entry"""
        log_entry = {}
        log_entry['log_type'] = log_type
        log_entry['logged_date'] = datetime.datetime.utcnow()
        log_entry['message'] = message
        if (inspect.stack()[3][3]):
            # If the call stack is changed the first array selector has to be changed
            log_entry['called_by'] = inspect.stack()[3][3]
        else:
            log_entry['called_by'] = 'Unknown caller'

        return log_entry

    def store_logger_names(self, name):
        #print('Appending name ', name)

        return self.loggers.append(name)

    # Definition of LogLevels, the lower the number, the stronger the logging will be
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

    logger.debug('Test-Log debug message')
    logger.info('Test-Log info message')
    logger.warn('Test-Log warn message')
    logger.error('Test-Log error message')

    # Starting here, only warn and error logs
    # should be created
    logger.set_verbosity(0)
    logger.debug('Test-Log debug message')
    logger.info('Test-Log info message')
    logger.warn('Test-Log warn message')
    logger.error('Test-Log error message')