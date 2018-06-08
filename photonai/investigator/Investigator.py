from photonai.base.PhotonBase import Hyperpipe
from photonai.photonlogger.Logger import Singleton, Logger
from photonai.investigator.app.main import app
from photonai.investigator.app.controller.helper import shutdown_server
from threading import Thread
from multiprocessing import Process
import webbrowser
import os
from time import sleep as slp

class Investigator:

    @staticmethod
    def build_url(storage: str, name: str):
        url = "http://localhost:7273/pipeline/" + storage + "/" + name
        return url

    @staticmethod
    def show(pipe: Hyperpipe):

        assert isinstance(pipe, Hyperpipe), "Investigator.show needs an object of type Hyperpipe"
        assert pipe is not None, "Investigator.show needs an object of Hyperpipe, is None"
        assert pipe.result_tree is not None, "Investigator.show needs an Hyperpipe that is already optimized, so it can show the result tree"
        # make sure that Flask is running
        FlaskManager().set_pipe_object(pipe.name, pipe.result_tree)
        url = Investigator.build_url("a", pipe.name)
        Investigator.delayed_browser(url)
        FlaskManager().run_app()

    @staticmethod
    def load_from_db(mongo_connect_url: str, pipe_name: str):
        FlaskManager().set_mongo_db_url(mongo_connect_url)
        url = Investigator.build_url("m", pipe_name)
        Logger().info("Your url is: " + url)
        Investigator.delayed_browser(url)
        FlaskManager().run_app()


    @staticmethod
    def load_many_from_db(mongo_connect_url: str, pipe_names: list):
        FlaskManager().set_mongo_db_url(mongo_connect_url)
        for pipe in pipe_names:
            url = Investigator.build_url("m", pipe)
            Logger().info("Your url is: " + url)
        FlaskManager().run_app()

    @staticmethod
    def load_from_file(name: str, file_url: str):
        assert os.path.isfile(file_url), "File" + file_url + " does not exist or is not a file. Please give absolute path."
        FlaskManager().set_pipe_file(name, file_url)
        url = Investigator.build_url("f", name)
        Investigator.delayed_browser(url)
        FlaskManager().run_app()

    @staticmethod
    def load_files(file_list: list):
        for file_url in file_list:
            Investigator.load_from_file(file_url)

    @staticmethod
    def open_browser(url):
        # we delay the browser opening for 2 seconds so that flask server can start in the meanwhile
        slp(2)
        webbrowser.open(url)

    @staticmethod
    def delayed_browser(url):
        Investigator.open_browser(url)
        # thread = Thread(target=Investigator.open_browser, args=(url, ))
        # thread.start()
        # thread.join()


@Singleton
class FlaskManager:

    def __init__(self):
        pass

    def set_mongo_db_url(self, mongo_url):
        app.config['mongo_db_url'] = mongo_url

    def set_pipe_file(self, name, path):
        app.config['pipe_files'][name] = path

    def set_pipe_object(self, name, obj):
        app.config['pipe_objects'][name] = obj

    def run_app(self):
        try:
            app.run(host='0.0.0.0', port=7273)
        except OSError as exc:
            if exc.errno == 98:
                # app already running
                pass
            else:
                raise exc

    # def run_app_in_process(self):
    #
    #     server = Process(target=FlaskManager().run_app())
    #     server.start()
    #     # server.terminate()
    #     # server.join()
    #
    # def __exit__(self, exc_type, exc_value, traceback):
    #     shutdown_server()
