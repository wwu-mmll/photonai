import configparser
from pathlib import Path
import datetime

class PhotonConf:
    configfile_path = "../configuration.ini"

    def create_default_config(self):
        self.config['LOGGING'] = {
            'print_to_console': True,
            'print_to_file': True,
            'logfile_name': str('photon_' + str(datetime.datetime.utcnow()) + '.log'),
            'print_to_slack': False,
            'slack_token': '',
            'loglevel_slack': 'INFO',
            'slack_channel': '#photon-log'
        }
        with open(self.configfile_path, 'w') as configfile:
            self.config.write(configfile)

    def __init__(self):
        self.config = configparser.ConfigParser()
        configfile = Path(self.configfile_path)
        if configfile.is_file():
            # file exists
            print("Loading configuration file.")
            self.config.read(self.configfile_path)
        else:
            print("Creating configuration file.")
            self.create_default_config()


