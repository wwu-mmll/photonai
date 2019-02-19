

class BestConfigTrace:
    """ Class stores metric lists to print them for plot traces
    author: Julian Gebker
    version: 1.0.0
    """

    def __init__(self, trace_name: str, metric_list: list=None, trace_mode: str='markers', trace_type: str='scatter'):
        """
        Constructor
        :param trace_name: Variable name and displayed name
        :param metric_list: List of metrics
        :param trace_mode: Trace's mode (default: markers)
        :param trace_type: Trace's type (default: scatter)
        """

        self.trace_name = trace_name
        if metric_list is None:
            self.metric_list = []
        else:
            self.metric_list = metric_list
        self.trace_mode = trace_mode
        self.trace_type = trace_type

    def values_x(self):
        """ function to print all metric names comma separated
        :return: Comma separated string of x values
        """
        result = ""
        for item in self.metric_list:
            result += "'" + str(item.name) + "', "
        return result

    def values_y(self):
        """ function to print all metric values comma separated
        :return: Comma separated string of y values
        """
        result = ""
        for item in self.metric_list:
            result += str(item.value) + ", "
        return result
