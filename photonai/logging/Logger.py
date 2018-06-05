import datetime
import inspect
from enum import Enum
from functools import total_ordering

from slackclient import SlackClient

from photonai.configuration.PhotonConf import PhotonConf

""" logging is a simple way to emit and store logs.

    The default LogLevel is WARN. It should only be increased 
    (to info or debug) if you need more detailed information,
    because extensive logging significantly impacts performance
    and clutters the database. The logs can also be printed on
    the console by setting print_to_console to True.
    
    Usage: 
    1) Import with
        from logging import Logger
    2) Log with
        logger = Logger()
        logger.debug('logging message!')
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

    def __call__(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class Logger:
    def __init__(self):
        # load configuration
        conf = PhotonConf()
        logging_conf = conf.config['LOGGING']

        # handle multiple instances of hyperpipe
        self.loggers = []

        # Set default LogLevel
        # Should be LogLevel.WARN!
        # Is LogLevel.DEBUG now only for testing purposes
        self._log_level = self.LogLevel.INFO
        self.verbosity_level = 0

        self.log_level_console = self.LogLevel.INFO
        self.log_level_slack = self.LogLevel.INFO
        self.log_level_file = self.LogLevel.INFO

        # Should the log also be printed to the console?
        # Recommendation: Set to True during development, false in production-environment
        self._print_to_console = logging_conf.getboolean('print_to_console')
        self._print_to_slack = logging_conf.getboolean('print_to_slack')
        self._slack_token = logging_conf['slack_token']
        self._slack_channel = logging_conf['slack_channel']
        self._print_to_file = logging_conf.getboolean('print_to_file')
        self._logfile_name = logging_conf['logfile_name']
        with open(self._logfile_name, "w") as text_file:
            text_file.write('PHOTON LOGFILE - ' + str(datetime.datetime.utcnow()))

    @staticmethod
    def set_print_to_console(self, status: bool):
        self._print_to_console = status

    def set_print_to_slack(self, status: bool):
        self._print_to_slack = status

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
    # Handles possible console-logging
    def _distribute_log(self, message: str, log_type: str):

        entry = self._generate_log_entry(message, log_type)

        if self._print_to_console:
            self._print_entry(entry)
        if self._print_to_file:
            self._write_to_file(entry)
        if self._print_to_slack:
            self._send_to_slack(entry)

    def _send_to_slack(self, entry: dict):
        if self._slack_token:
            try:
                sc = SlackClient(self._slack_token)

                sc.api_call(
                    "chat.postMessage",
                    channel=self._slack_channel,
                    text="{}: {}".format(entry['log_type'], entry['message'])
                )
            except:
                # Todo: catch channel not found exception
                print("Could not print to Slack") # <- cant use Logger here because it would cause an endless loop
                pass
        else:
            print('Error: No Slack Token Set')

    @staticmethod
    def _print_entry(entry: dict):
        date_str = entry['logged_date'].strftime("%Y-%m-%d %H:%M:%S")
        #print("{0} UTC - {1}: {2}".format(date_str, entry['log_type'], entry['message']))
        print("{0}".format(entry['message']))

    def _write_to_file(self, entry: dict):
        with open(self._logfile_name, "a", newline='\n') as text_file:
            text_file.write('\n')
            text_file.write(str(entry['message']))

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
