

class PlotlyTrace:
    """ Class represents a trace made with Plotly
    author: Julian Gebker
    version: 1.0.0
    """

    def __init__(self, variable_name: str, mode: str="markers", trace_type: str="scatter", trace_size: int=0, trace_color: str="", with_error: bool=False):
        """ Constructor
        :param variable_name: Variable name in javascript
        :param mode: Trace's mode (default: 'markers')
        :param trace_type: Trace's type (default: 'scatter')
        """
        self.variable_name = variable_name
        self.mode = mode
        self.trace_type = trace_type
        self.x = []
        self.y = []
        self.error_y = []
        self.trace_size = trace_size
        self.with_error = with_error
        self.trace_color = trace_color

    def add_x(self, x):
        """ function to add new value for x axis
        :param x: value for x axis
        """
        self.x.append(x)

    def add_y(self, y):
        """ function to add new value for y axis
        :param y: value for y axis
        """
        self.y.append(y)

    def add_error_y(self, y):
        """ function to add new value for y axis
        :param y: add y error value
        """
        self.error_y.append(y)

    def get_x_to_string(self) -> str:
        """ function to print all x values comma separated
        :return: comma separated string of x values
        """
        result = ""
        for item in self.x:
            result += "'" + str(item) + "',"
        return result.rstrip(",")

    def get_y_to_string(self) -> str:
        """ function to print all y values comma separated
        :return: comma separated string of y values
        """
        result = ""
        for item in self.y:
            result += "'" + str(item) + "',"
        return result.rstrip(",")

    def get_err_y_to_string(self) -> str:
        """ function to print all err_y values comma separated
        :return: comma separated string of y error values
        """
        result = ""
        for item in self.error_y:
            result += "'" + str(item) + "',"
        return result.rstrip(",")
