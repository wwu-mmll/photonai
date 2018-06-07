

class ConfigItem:
    """ Representing a Config Dict Item with name and value
    author: Julian Gebker
    version: 1.0.0
    """

    def __init__(self, name: str, value: str):
        """ Constructor
        :param name: Config name
        :param value: Config value
        """
        self.name = name
        self.value = value
