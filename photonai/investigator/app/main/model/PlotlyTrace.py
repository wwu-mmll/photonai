import numpy as np


class PlotlyTrace:
    """ Class represents a trace made with Plotly
    author: Julian Gebker
    version: 1.0.0
    """
    COLOR_PALETTE = {
        # "train_color": "#379683",
        # "train_color_bold": "#00624E",
        # "val_color": "#FFA347",
        # "val_color_bold": "#CF6700",
        # "test_color": "#FF8045",
        # "test_color_bold": "#E94A00",
        # "dummy_color": "#DEDEDE",
        # "dummy_color_bold": "#BCBCBC",
        "train_color": "#2388FE",
        "train_color_bold": "#0E57AB",
        "val_color": "#C7D7FE",
        "val_color_bold": "#0E0E1D",
        "test_color": "#5C4A79",
        "test_color_bold": "#0E0E1D",
        "dummy_color": "#DEDEDE",
        "dummy_color_bold": "#BCBCBC",
        "alternative_test_color": "#FF5930",
        "alternative_test_color_bold": "#E22D00"
    }

    def __init__(self, variable_name: str, mode: str="markers", trace_type: str="scatter", trace_size: int=0,
                 trace_color: str="", with_error: bool=False, colorscale: list=None, marker_line_width: int=None,
                 marker_line_color: str = None, width=None, opacity: float = 1.):
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
        self.z = []
        self.error_y = []
        self.trace_size = trace_size
        self.with_error = with_error
        self.trace_color = self.get_color(trace_color)
        self.colorscale = colorscale
        self.marker_line_width = marker_line_width
        self.marker_line_color = marker_line_color
        self.width = width
        self.opacity = opacity

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

    def add_z(self, z):
        """ function to add new value for z axis
                :param z: value for z axis
                """
        self.z.append(z)

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
            if isinstance(item, list):
                result += "["
                for iitem in item:
                    result += "'" + str(iitem) + "',"
                result.rstrip(",")
                result += "],"
            else:
                result += "'" + str(item) + "',"
        return result.rstrip(",")

    def get_y_to_string(self) -> str:
        """ function to print all y values comma separated
        :return: comma separated string of y values
        """
        result = ""
        for item in self.y:
            if isinstance(item, list):
                result += "["
                for iitem in item:
                    result += "'" + str(iitem) + "',"
                result.rstrip(",")
                result += "],"
            else:
                result += "'" + str(item) + "',"
        return result.rstrip(",")

    def get_err_y_to_string(self) -> str:
        """ function to print all err_y values comma separated
        :return: comma separated string of y error values
        """
        result = ""
        for item in self.error_y:
            if isinstance(item, list):
                result += "["
                for iitem in item:
                    result += "'" + str(iitem) + "',"
                result.rstrip(",")
                result += "],"
            else:
                result += "'" + str(item) + "',"
        return result.rstrip(",")

    def get_z_to_string(self, as_numbers: bool = False) -> str:
        result = ""
        for item in self.z:
            if isinstance(item, (list, np.ndarray)):
                result += "["
                for iitem in item:
                    if as_numbers:
                        result += str(iitem) + ","
                    else:
                        result += "'" + str(iitem) + "',"
                result = result.rstrip(",")
                result += "],"
            else:
                if as_numbers:
                    result += "'" + str(item) + "',"
                else:
                    result += str(item) + ","
        return result.rstrip(",")

    def get_color(self, color):
        if color in self.COLOR_PALETTE.keys():
            return self.COLOR_PALETTE[color]
        else:
            return color
