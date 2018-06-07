from photonai.base.PhotonBase import Hyperpipe
from photonai.photonlogger.Logger import Singleton
from photonai.investigator.app.main import app


class Investigator:

    @staticmethod
    def show(pipe: Hyperpipe):

        # make sure that Flask is running
        FlaskManager().set_config_item("pipe_3", pipe.result_tree)
        FlaskManager().run_app()

    @staticmethod
    def load_from_db(mongo_connect_url: str, pipe_name: str):
        pass

    @staticmethod
    def load_from_file(file_url: str):
        pass


@Singleton
class FlaskManager:

    def __init__(self):
        pass

    def set_config_item(self, name, obj):
        app.config[name] = obj

    def run_app(self):
        app.run(host='0.0.0.0', port=7273)
