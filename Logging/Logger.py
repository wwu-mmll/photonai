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


# Create the photon_log_db
client = MongoClient('localhost', 27017)
log_db = client.your_collection


# All collections should probably be capped so that no manual log-pruning is necessary
# like info_log_collection = log_db.createCollection("info_log", capped=True, size=15000)?!
debug_log_collection = log_db.debug_log
debug_log_collection.ensure_index([('logged_date', ASCENDING)])

verbose_log_collection = log_db.verbose_log
verbose_log_collection.ensure_index([('logged_date', ASCENDING)])

info_log_collection = log_db.info_log
info_log_collection.ensure_index([('logged_date', ASCENDING)])

warn_log_collection = log_db.warn_log
warn_log_collection.ensure_index([('logged_date', ASCENDING)])

error_log_collection = log_db.error_log
error_log_collection.ensure_index([('logged_date', ASCENDING)])



# Definition of LogLevels, the higher the number, the stronger the logging will be
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


# Set default LogLevel
# Should be LogLevel.WARN!
# Is LogLevel.DEBUG now only for testing purposes
_log_level = LogLevel.DEBUG


# Should the log also be printed to the console?
# Recommendation: Set to True during development, false in production-environment
_print_to_console = True


def set_print_to_console(status: bool):
    _print_to_console = status


def set_log_level(level: LogLevel):
    """" Use this method to change the log level. """
    _log_level = level


# Debug should be used for information that may be useful for program-debugging (most information)
# i.e. training epochs of neural nets
def debug(message: str):
    if(_log_level <= LogLevel.DEBUG):
        _insert_log_into_database(message, debug_log_collection, 'DEBUG')

# Verbose should be used if something interesting (but uncritically) happened
# i.e. every hp config that is tested
def verbose(message: str):
    if (_log_level <= LogLevel.VERBOSE):
        _insert_log_into_database(message, info_log_collection, 'VERBOSE')


# Info should be used if something interesting (but uncritically) happened
# i.e. most basic info on photon hyperpipe
def info(message: str):
    if(_log_level <= LogLevel.INFO):
        _insert_log_into_database(message, info_log_collection, 'INFO')


# Something may have gone wrong? Use warning
def warn(message: str):
    if (_log_level <= LogLevel.WARN):
        _insert_log_into_database(message, warn_log_collection, 'WARN')


# Something broke down. Error should be used if something unexpected happened
def error(message: str):
    if (_log_level <= LogLevel.ERROR):
        _insert_log_into_database (message, error_log_collection, 'ERROR')


# Takes a message and inserts it into the given collection
# Handles possible console-logging
def _insert_log_into_database(message: str, collection, log_type: str):
    entry = _generate_log_entry(message, log_type);
    collection.insert(entry)

    if (_print_to_console):
        _print_entry(entry);


def _print_entry(entry: dict):
    print ('['+entry['logged_date'].strftime('%d %b %Y - %H:%M:%S.%f')+']' +
           ' ' + entry['called_by'] +
           ' ' + entry['log_type'] +
           ' ' + entry['message'])


def _generate_log_entry(message: str, log_type: str):
    """Todo: Get current user from user-service and add username to log_entry"""
    log_entry = {}
    log_entry['log_type'] = log_type
    log_entry['logged_date'] = datetime.datetime.utcnow()
    log_entry['message'] = message
    if (inspect.stack()[3][3]):
        # If the call stack is changed the first array selector has to be changed
        log_entry['called_by'] = inspect.stack()[3][3];
    else:
        log_entry['called_by'] = 'Unknown caller'

    return log_entry


if __name__ == "__main__":
    set_log_level(LogLevel.DEBUG)

    debug('Test-Log debug message')
    info('Test-Log info message')
    warn('Test-Log warn message')
    error('Test-Log error message')

    # Starting here, only warn and error logs
    # should be created
    set_log_level(LogLevel.WARN)
    debug('Test-Log debug message')
    info('Test-Log info message')
    warn('Test-Log warn message')
    error('Test-Log error message')