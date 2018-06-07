

class Metric:
    """Representing a PHOTON metric with name and value
    author: Julian Gebker
    version: 1.0.0
    """

    def __init__(self, name: str, value: str):
        """ Constructor
        :param name: Metric's name
        :param value: Metric's value
        """
        self.name = name
        self.value = value
