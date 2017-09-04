import configparser
from pathlib import Path

class PhotonConf:
    configfile_path = "../configuration.ini"


    def createDefaultConfig(self):
        self.config['LOGGING'] = {
            'print_to_console': True,
            'print_to_file': True,
            'logfile_name' : 'photon_out.log',
            'print_to_slack': False,
            'slack_token': 'enter token here!',
            'loglevel_slack': 'INFO'
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
            self.createDefaultConfig()


## Tests
configtest = PhotonConf()
print(configtest.config['LOGGING']['logfile_name'])
