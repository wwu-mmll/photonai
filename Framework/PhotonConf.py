import configparser
from pathlib import Path

class PhotonConf:
    configfile_name = "configuration.ini"


    def createDefaultConfig(self):
        self.config['LOGGING'] = {
            'print_to_console': True,
            'print_to_file': True,
            'logfile_name' : 'photon_out.log',
            'print_to_slack': False,
            'slack_token': {},
            'loglevel_slack': 'INFO'
        }
        with open(self.configfile_name, 'w') as configfile:
            self.config.write(configfile)

    def __init__(self):

        self.config = configparser.ConfigParser()
        configfile = Path(self.configfile_name)
        if configfile.is_file():
            # file exists
            self.config.read(self.configfile_name)
        else:
            print("Creating configuration file")
            self.createDefaultConfig()


