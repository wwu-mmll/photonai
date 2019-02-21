from ..model.ConfigItem import ConfigItem


class Config:
    """ Representing a Config Dict with name
    author: Julian Gebker
    version: 1.0.0
    """

    def __init__(self, name: str, config_nr: int=None, items: list=None):
        """ Constructor
        :param name: Config name
        :param config_nr: Config number
        :param items: Config items
        """
        if items is None:
            self.items = []
        else:
            self.items = items
        self.name = name
        self.config_nr = config_nr

    def add_item(self, item: ConfigItem):
        """ Function adds given item to self.items
        :param item: Item to add
        """
        self.items.append(item)
