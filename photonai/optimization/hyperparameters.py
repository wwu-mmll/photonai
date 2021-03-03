import warnings
import random
import numpy as np

from photonai.photonlogger.logger import logger


class PhotonHyperparam(object):
    """Photon hyperparameter.

    A base class that manages its own range of values.

    """
    def __init__(self, values: list):
        """
        Initialize the object.

        Parameters:
            values:
                Parameter Domain.

        """
        self.values = values

    def get_random_value(self, definite_list: bool = True):
        """
        Method for random search to get a random value based on the underlying domain.

        Parameters:
            definite_list:
                 Choice  between an element of a discrete list or a value within an interval.
                 In some cases, certain settings such as the
                 step size may otherwise be lost.

        """
        if definite_list:
            return random.choice(self.values)
        else:
            msg = "The PHOTONAI hyperparam has no own random function."
            logger.error(msg)
            raise NotImplementedError(msg)

    def __str__(self):
        return str(self.__class__) + str(self.values)


class Categorical(PhotonHyperparam):
    """
    Class for defining a definite list of values.

    """
    def __init__(self, values: list):
        """
        Initialize the object.

        Parameters:
            values:
                Definite list of hyperparameter values.

        """
        super(Categorical, self).__init__(values)

    def __getitem__(self, item):
        return self.values.__getitem__(item)


class BooleanSwitch(PhotonHyperparam):
    """Boolean switch.

    Class for defining a boolean hyperparameter.
    Equivalent to Categorical([True, False]).

    """
    def __init__(self):
        """Initialize the object."""
        super(BooleanSwitch, self).__init__([True, False])


class NumberRange(PhotonHyperparam):
    """Number range.

    Class for easily creating a range of numbers to be tested in the optimization process.

    Notes:

        Before the values of the domain are available,
        it is mandatory to call the transform method.

    """
    def __init__(self, start: float, stop: float, range_type: str, step: int = None,
                 num: int = None, num_type: type = np.int64, **kwargs):
        """
        Initialize the object.

        Parameters:
            start:
                The start value for generating the lower bound.
                The resulting interval includes the value.

            stop:
                The stop value for generating the upper bound.

                - if range_type == "range":
                    The end value is not included in the interval (see documentation of numpy.arange).
                - if range_type == "linspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.linspace).
                - if range_type == "logspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.logspace).
                - if range_type == "geomspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.logspace).

            range_type:
                Which method to use for generating the number interval.
                Possible options:

                - "range": numpy.arange is used to generate a list
                    of values separated by the same step width.
                - "linspace": numpy.linspace is used to generate a certain
                    number of values between start and stop.
                - "logspace": numpy.logspace is used to generate a logarithmically
                    distributed range of a certain length.
                - "geomspace": numpy.geomspace is used to generate numbers spaced
                    evenly on a log scale (geometric progression).

            num_type:
                The underlying datatype of the values.
                For the inheriting class IntegerRange it is set to np.int32.
                For the inheriting class FloatRange it is set to np.float32.

            step:
                if range_type == 'range', the spacing between values.

            num:
                if range_type == 'linspace', range_type == 'logspace', or range_type == 'geomspace',
                the number of samples to generate.

            **kwargs:
                Further parameters that should be passed to the numpy function chosen with range_type.

        """
        super(NumberRange, self).__init__([])
        self.start = start
        self.stop = stop
        self._range_type = None
        self.range_type = range_type
        self.range_params = kwargs
        self.num_type = num_type
        self.step = step
        self.num = num

    def transform(self):
        """
        Translates the definition into an area with values.
        These values are again stored in the attribute self.values.

        """
        if self.range_type == "geomspace" and self.start == 0:
            error_message = "Geometric sequence cannot include zero"
            logger.error(error_message)
            raise ValueError(error_message)
        if self.range_type == "range" and self.start > self.stop:
            warn_message = "NumberRange or one of its subclasses is empty cause np.arange " + \
                           "does not deal with start greater than stop."
            logger.warning(warn_message)
            warnings.warn(warn_message)

        values = []

        if self.range_type == "range":
            if not self.step:
                values = np.arange(self.start, self.stop, dtype=self.num_type, **self.range_params)
            else:
                values = np.arange(self.start, self.stop, self.step, dtype=self.num_type, **self.range_params)
        elif self.range_type == "linspace":
            if self.num:
                values = np.linspace(self.start, self.stop, num=self.num, dtype=self.num_type, **self.range_params)
            else:
                values = np.linspace(self.start, self.stop, dtype=self.num_type, **self.range_params)
        elif self.range_type == "logspace":
            if self.num_type == np.int32:
                raise ValueError("Cannot use logspace for integer,  use geomspace instead.")
            if self.num:
                values = np.logspace(self.start, self.stop, num=self.num, dtype=self.num_type, **self.range_params)
            else:
                values = np.logspace(self.start, self.stop, dtype=self.num_type, **self.range_params)
        elif self.range_type == "geomspace":
            if self.num:
                values = np.geomspace(self.start, self.stop, num=self.num, dtype=self.num_type, **self.range_params)
            else:
                values = np.geomspace(self.start, self.stop, dtype=self.num_type, **self.range_params)

        # convert to python datatype because mongodb needs it
        try:
            self.values = [values[i].item() for i in range(len(values))]
        except:
            msg = "PHOTON can not guarantee full mongodb support since you chose a non [np.integer, np.floating] " \
                  "subtype in NumberType.dtype."
            logger.warning(msg)
            warnings.warn(msg)
            self.values = values

    @property
    def range_type(self):
        return self._range_type

    @range_type.setter
    def range_type(self, value):
        range_types = ["range", "linspace", "logspace", "geomspace"]
        if value in range_types:
            self._range_type = value
        else:
            raise ValueError("Subclass of NumberRange supports only "+str(range_types)+" as range_type, not " +
                             repr(value))


class IntegerRange(NumberRange):
    """Integer range.

    Class for easily creating a range of integers
    to be tested in optimization process.

    """
    def __init__(self, start: float, stop: float, range_type: str = 'range',
                 step: int = None, num: int = None, **kwargs):
        """
        Initialize the object.

        Parameters:
            start:
                The start value for generating the lower bound.
                The resulting interval includes the value.

            stop:
                The stop value for generating the upper bound.

                - if range_type == "range":
                    The end value is not included in the interval (see documentation of numpy.arange).
                - if range_type == "linspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.linspace).
                - if range_type == "logspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.logspace).
                - if range_type == "geomspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.logspace).

            range_type:
                Which method to use for generating the number interval.
                Possible options,

                - "range": numpy.arange is used to generate a list
                    of values separated by the same step width.
                - "linspace": numpy.linspace is used to generate a certain
                    number of values between start and stop.
                - "logspace": numpy.logspace is used to generate a logarithmically
                    distributed range of a certain length.
                - "geomspace": numpy.geomspace is used to generate numbers spaced
                    evenly on a log scale (geometric progression).

            step:
                If range_type == 'range', the spacing between values.

            num:
                If range_type == 'linspace', range_type == 'logspace', or range_type == 'geomspace',
                the number of samples to generate.

            **kwargs:
                Further parameters that should be passed to the numpy function chosen with range_type.

        """
        super().__init__(start, stop, range_type, step, num, np.int32, **kwargs)

    def get_random_value(self, definite_list: bool = False):
        """
        Method for random search to get a random value based on the underlying domain.

        Parameters:
            definite_list:
                 Choice  between an element of a discrete list or a value within an interval.
                 As example, the step parameter would vanishes when this parameter
                 is set to False.

        """
        if definite_list:
            if not self.values:
                msg = "No values were set. Please use transform method."
                logger.error(msg)
                raise ValueError(msg)
            return random.choice(self.values)
        else:
            return random.randint(self.start, self.stop-1)


class FloatRange(NumberRange):
    """Float range.

    Class for easily creating an interval of numbers
    to be tested in the optimization process.

    """
    def __init__(self, start: float, stop: float, range_type: str = 'linspace',
                 step: float = None, num: int = None, **kwargs):
        """
        Initialize the object.

        Parameters:
            start:
                The start value for generating the lower bound.
                The resulting interval includes the value.

            stop:
                The stop value for generating the upper bound.

                - if range_type == "range":
                    The end value is not included in the interval (see documentation of numpy.arange).
                - if range_type == "linspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.linspace).
                - if range_type == "logspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.logspace).
                - if range_type == "geomspace"
                    The end value is included in the interval,
                    unless endpoint is set to False (see documentation of numpy.logspace).

            range_type:
                Which method to use for generating the number interval.
                Possible options,

                - "range": numpy.arange is used to generate a list
                    of values separated by the same step width.
                - "linspace": numpy.linspace is used to generate a certain
                    number of values between start and stop.
                - "logspace": numpy.logspace is used to generate a logarithmically
                    distributed range of a certain length.
                - "geomspace": numpy.geomspace is used to generate numbers spaced
                    evenly on a log scale (geometric progression).

            step:
                If range_type == 'range', the spacing between values.

            num:
                If range_type == 'linspace', range_type == 'logspace', or range_type == 'geomspace',
                the number of samples to generate.

            kwargs:
                Further parameters that should be passed to the numpy function chosen with range_type.

        """
        super(FloatRange, self).__init__(start, stop, range_type, step, num, np.float64, **kwargs)

    def get_random_value(self, definite_list: bool = False):
        """
        Method for random search to get a random value based on the underlying domain.

        Parameters:
            definite_list:
                 Choice  between an element of a discrete list or a value within an interval.
                 As example, the num parameter would vanishes when this parameter
                 is set to False.

        """
        if definite_list:
            if not self.values:
                msg = "No values were set. Please use transform method."
                logger.error(msg)
                raise ValueError(msg)
            return random.choice(self.values)
        else:
            return random.uniform(self.start, self.stop)
