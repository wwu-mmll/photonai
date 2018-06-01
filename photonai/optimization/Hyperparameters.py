import numpy as np


class ValueList:
    """
      Class for defining a  definite list of hyperparameter values.
      Can be used for categorical values, but also for numbers.

      Parameters
      ----------
      * 'values' [list]:
         definite list of hyperparameter values

    """

    def __init__(self, values: list):
        self.values = values


class BooleanSwitch:
    """
      Class for defining a boolean hyperparameter, when both options should be tested in hyperparameter optimization.

      Parameters
      ----------
      * 'values' [bool]:
         will return both True, and False

    """

    def __init__(self):
        self.values = [True, False]


class NumberRange:
    """
      Class for easily creating a range of numbers to be tested in hyperparameter optimization.

      Parameters
      ----------
      * 'start' [number]:
         The start value for generating the number interval.
         The resulting interval includes the value, default is 0.

      * 'stop' [number]:
         The stop value for generating the number interval.

         - if range_type == "range":
           the end value is not included in the interval (see documentation of numpy.arange).
         - if range_type == "linspace"
           the end value is included in the interval, unless endpoint is set to False (see documentation of numpy.linspace).
        - if range_type == "logspace"
           the end value is included in the interval, unless endpoint is set to False (see documentation of numpy.logspace).

      * 'range_type' [str]:
         Which method to use for generating the number interval.
         Possible options:

         - "range": numpy.arange is used to generate a list of values separated by the same step width.
         - "linspace": numpy.linspace is used to generate a certain number of values between start and stop.
         - "logspace": numpy.logspace is used to generate a logarithmically distributed range of a certain length.


      * 'num_type' [numpy.dtype]:
         The specific type specification for the interval's numbers.

         For the inheriting class IntegerRange it is set to np.int32.
         For the inheriting class FloatRange it is set to np.float32.

      * 'step' [number, default=None, optional]:
        if range_type == 'range', the spacing between values.

      * 'num' [int, default=None, optional]:
        if range_type == 'linspace' or range_type == 'logspace', the number of samples to generate.

      * 'kwargs' [dict, optional]:
        Further parameters that should be passed to the numpy function chosen with range_type.
    """

    def __init__(self, start, stop, range_type, num_type, step=None, num=None, **kwargs):

        self.start = start
        self.stop = stop
        self.type = range_type
        self.range_params = kwargs
        self.num_type = num_type
        self.values = self.transform()

    def transform(self):

        if self.range_type == "range":
            return np.arange(self.start, self.stop, dtype=self.num_type, **self.range_params)
        elif self.range_type == "linspace":
            return np.linspace(self.start, self.stop, dtype=self.num_type, **self.range_params)
        elif self.range_type == "logspace":
            return np.logspace(self.start, self.stop, dtype=self.num_type, **self.range_params)
        else:
            return []


class IntegerRange(NumberRange):
    """
         Class for easily creating a range of integer numbers to be tested in hyperparameter optimization.

         Parameters
         ----------
         * 'start' [number]:
            The start value for generating the number interval.
            The resulting interval includes the value, default is 0.

         * 'stop' [number]:
            The stop value for generating the number interval.

            - if range_type == "range":
              the end value is not included in the interval (see documentation of numpy.arange).
            - if range_type == "linspace"
              the end value is included in the interval, unless endpoint is set to False (see documentation of numpy.linspace).
           - if range_type == "logspace"
              the end value is included in the interval, unless endpoint is set to False (see documentation of numpy.logspace).

         * 'range_type' [str]:
            Which method to use for generating the number interval.
            Possible options:

            - "range": numpy.arange is used to generate a list of values separated by the same step width.
            - "linspace": numpy.linspace is used to generate a certain number of values between start and stop.
            - "logspace": numpy.logspace is used to generate a logarithmically distributed range of a certain length.

         * 'step' [number, default=None, optional]:
           if range_type == 'range', the spacing between values.

         * 'num' [int, default=None, optional]:
           if range_type == 'linspace' or range_type == 'logspace', the number of samples to generate.

         * 'kwargs' [dict, optional]:
           Further parameters that should be passed to the numpy function chosen with range_type.
       """

    def __init__(self, start, stop, range_type, **kwargs):
            super().__init__(start, stop, range_type, np.int32, **kwargs)


class FloatRange(NumberRange):
    """
          Class for easily creating a range of integer numbers to be tested in hyperparameter optimization.

          Parameters
          ----------
          * 'start' [number]:
             The start value for generating the number interval.
             The resulting interval includes the value, default is 0.

          * 'stop' [number]:
             The stop value for generating the number interval.

             - if range_type == "range":
               the end value is not included in the interval (see documentation of numpy.arange).
             - if range_type == "linspace"
               the end value is included in the interval, unless endpoint is set to False (see documentation of numpy.linspace).
            - if range_type == "logspace"
               the end value is included in the interval, unless endpoint is set to False (see documentation of numpy.logspace).

          * 'range_type' [str]:
             Which method to use for generating the number interval.
             Possible options:

             - "range": numpy.arange is used to generate a list of values separated by the same step width.
             - "linspace": numpy.linspace is used to generate a certain number of values between start and stop.
             - "logspace": numpy.logspace is used to generate a logarithmically distributed range of a certain length.

          * 'step' [number, default=None, optional]:
            if range_type == 'range', the spacing between values.

          * 'num' [int, default=None, optional]:
            if range_type == 'linspace' or range_type == 'logspace', the number of samples to generate.

          * 'kwargs' [dict, optional]:
            Further parameters that should be passed to the numpy function chosen with range_type.
        """

    def __init__(self, start, stop, range_type, **kwargs):
            super().__init__(start, stop, range_type, np.float32, **kwargs)
