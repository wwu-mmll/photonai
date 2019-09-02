from ..base.PhotonBase import Hyperpipe, Stack, Branch, Switch
from ..photonlogger.Logger import Singleton, Logger
from ..investigator.app.main import application
from ..optimization.Hyperparameters import FloatRange, IntegerRange, Categorical

import webbrowser
import os
from time import sleep as slp
from threading import Thread


class Investigator:
    """
    Instantiates a Flask website that shows you the results of the hyperparameter search, the best configuration,
    all of its performances etc.
    """

    @staticmethod
    def __build_url(storage: str, name: str):
        """
        creates a localhost url for displaying a pipeline according to its source (working memory, file or mongodb)
        """
        url = "http://localhost:7275/pipeline/" + storage + "/" + name
        return url

    @staticmethod
    def show(pipe: Hyperpipe):
        """
        Opens the PHOTON investigator and shows the hyperpipe's hyperparameter search performance from working space

        Parameters
        ----------
        * 'pipe' [Hyperpipe]:
            The Hyperpipe object that has performed hyperparameter search

        """

        assert isinstance(pipe, Hyperpipe), "Investigator.show needs an object of type Hyperpipe"
        assert pipe is not None, "Investigator.show needs an object of Hyperpipe, is None"
        assert pipe.results is not None, "Investigator.show needs an Hyperpipe that is already optimized, so it can show the result tree"
        # make sure that Flask is running
        FlaskManager().set_pipe_object(pipe.name, pipe.results)
        url = Investigator.__build_url("a", pipe.name)
        Logger().info("Your url is: " + url)
        Investigator.__delayed_browser(url)
        FlaskManager().run_app()


    @staticmethod
    def load_from_db(mongo_connect_url: str, pipe_name: str):
        """
        Opens the PHOTON investigator and
        loads a hyperpipe's performance results from a MongoDB instance

        Parameters
        ---------
        * 'mongo_connect_url' [str]:
            The MongoDB connection string including the database name
        * 'pipe_name' [str]:
            The name of the pipeline to load
        """
        FlaskManager().set_mongo_db_url(mongo_connect_url)
        url = Investigator.__build_url("m", pipe_name)
        Logger().info("Your url is: " + url)
        Investigator.__delayed_browser(url)
        FlaskManager().run_app()


    @staticmethod
    def load_many_from_db(mongo_connect_url: str, pipe_names: list):
        """
        Opens the PHOTON investigator and
        loads a hyperpipe performance results from a MongoDB instance

        Parameters
        ---------
        * 'mongo_connect_url' [str]:
            The MongoDB connection string including the database name
        * 'pipe_names' [list]:
            A list of the hyperpipe objects to load
        """

        FlaskManager().set_mongo_db_url(mongo_connect_url)
        for pipe in pipe_names:
            url = Investigator.__build_url("m", pipe)
            Logger().info("Your url is: " + url)
        FlaskManager().run_app()

    @staticmethod
    def load_from_file(name: str, file_url: str):
        """
        Opens the PHOTON investigator and loads the hyperpipe search results from the file path

        Parameters
        ----------
        * 'name' [str]:
            The name of the hyperpipe object that you want to load
        * 'file_url' [str]:
            The path to the file in which the hyperparameter search results are encoded.
        """
        assert os.path.isfile(file_url), "File" + file_url + " does not exist or is not a file. Please give absolute path."
        FlaskManager().set_pipe_file(name, file_url)
        url = Investigator.__build_url("f", name)
        Investigator.__delayed_browser(url)
        FlaskManager().run_app()

    # @staticmethod
    # def load_files(file_list: list):
    #     """
    #        Opens the PHOTON investigator and loads the hyperpipe search results from the file path
    #
    #        Parameters
    #        ----------
    #        * 'file_url' [str]:
    #            The path to the file in which the hyperparameter search results are encoded.
    #     """
    #     for file_url in file_list:
    #         Investigator.load_from_file("" file_url)

    @staticmethod
    def __open_browser(url):
        # we delay the browser opening for 2 seconds so that flask server can start in the meanwhile
        slp(2)
        webbrowser.open(url)

    @staticmethod
    def __delayed_browser(url):
        # Investigator.__open_browser(url)
        thread = Thread(target=Investigator.__open_browser, args=(url, ))
        thread.start()
        thread.join()


@Singleton
class FlaskManager:

    def __init__(self):
        self.app = application

    def set_mongo_db_url(self, mongo_url):
        self.app.config['mongo_db_url'] = mongo_url

    def set_pipe_file(self, name, path):
        self.app.config['pipe_files'][name] = path

    def set_pipe_object(self, name, obj):
        self.app.config['pipe_objects'][name] = obj

    def run_app(self):
        try:
            self.app = application.run(host='0.0.0.0', port=7275)
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




class Flowchart(object):

    def __init__(self, hyperpipe):
        self.hyperpipe = hyperpipe
        self.pipeline_elements = hyperpipe.pipeline_elements
        self.chart_str = ""

    def create_str(self):

        #headerLayout = "[CrossValidation][Optimizer]\n\n[Input]"
        headerLayout = ""
        headerRelate = ""
        #oldElement = "Input"
        oldElement = ""
        for pipelineElement in self.pipeline_elements:
            headerLayout = headerLayout+"[" + pipelineElement.name +"]"
            if oldElement:
                headerRelate = headerRelate+"["+oldElement+"]"+"->"+"["+pipelineElement.name+"]\n"
            oldElement = pipelineElement.name

        # headerLayout += "[Output]"
        # headerRelate += "[" + oldElement + "]->[Output]\n"
        self.chart_str = "Layout:\n"+headerLayout+"\nRelate:\n"+headerRelate+"\n"
        # self.chart_str += "\n\n[Input]:\nDefine:\nX\n  shape: {}\n|\ny\n  shape: {}\n|\n**kwargs\n  shape: {}\n".format(self.hyperpipe.data.X.shape,
        #                                                                                                            self.hyperpipe.data.y.shape,
        #                                                                                                            'None')
        # self.chart_str += "[Output]:\nDefine:\n y_pred\n"

        for pipelineElement in self.pipeline_elements:
            self.chart_str = self.chart_str+self.recursivElement(pipelineElement,"")

        # # Add define for input and output
        # self.chart_str += "\n\n[CrossValidation]:\nDefine:\nOuterCV: {}\nInnerCV: {}\n".format(self.format_cross_validation(self.hyperpipe.cross_validation.outer_cv),
        #                                                                                        self.format_cross_validation(self.hyperpipe.cross_validation.inner_cv))
        # opt = self.format_optimizer(self.hyperpipe.optimization)
        # self.chart_str += "\n\n[Optimizer]:\nDefine:\noptimizer: {}\nOptimizerParams: {}\nMetrics: {}\nBest Config Metric: {}\n".format(opt[0], opt[1], opt[2], opt[3])

        return self.chart_str

    def format_cross_validation(self, cv):
        if cv:
            string = "{}(".format(cv.__class__.__name__)
            for key, val in cv.__dict__.items():
                string += "{}={}, ".format(key, val)
            return string[:-2] + ")"
        else:
            return "None"

    def format_optimizer(self, optimizer):
        return optimizer.optimizer_input, optimizer.optimizer_params, optimizer.metrics, optimizer.best_config_metric

    def format_kwargs(self, kwargs):
        pass

    def format_hyperparameter(self, hyperparameter):
        if isinstance(hyperparameter, IntegerRange):
            return """IntegerRange(start: {},
                                   stop: {}, 
                                   step: {}, 
                                   range_type: {})""".format(hyperparameter.start, hyperparameter.stop,
                                                                           hyperparameter.step, hyperparameter.range_type)
        elif isinstance(hyperparameter, FloatRange):
            return """FloatRange(start: {},
                                   stop: {}, 
                                   step: {}, 
                                   range_type: {})""".format(hyperparameter.start,
                                                               hyperparameter.stop,
                                                               hyperparameter.step,
                                                               hyperparameter.range_type)
        elif isinstance(hyperparameter, Categorical):
            return str(hyperparameter.values)
        else:
            return str(hyperparameter)

    def recursivElement(self,pipe_element,parent):

        string = ""
        if hasattr(pipe_element, "named_steps"):
            elements = list()
            for name, element in pipe_element.elements:
                elements.append(element)
            pipe_element.pipe_elements = elements

        if not hasattr(pipe_element, "named_steps"):
            if parent == "":
                string = "["+pipe_element.name+"]:\n"+"Define:\n"
            else:
                string = "[" + parent[1:]+"."+pipe_element.name + "]:\n" + "Define:\n"
            hyperparameters = None
            kwargs = None
            if hasattr(pipe_element, "hyperparameters"):
                hyperparameters = pipe_element.hyperparameters
                for name, parameter in pipe_element.hyperparameters.items():
                    string += "{}: {}\n".format(name.split('__')[-1], self.format_hyperparameter(parameter))
            if hasattr(pipe_element, "kwargs"):
                kwargs = pipe_element.kwargs
                for name, parameter in pipe_element.kwargs.items():
                    string += "{}: {}\n".format(name.split('__')[-1], self.format_hyperparameter(parameter))
            if not kwargs and not hyperparameters:
                string += "default\n"

        # Pipeline Stack
        elif isinstance(pipe_element, Stack):
            if parent == "":
                string = "[" + pipe_element.name + "]:\n" + "Layout:\n"
            else:
                string = "["+parent[1:]+"."+pipe_element.name+"]:\n"+"Layout:\n"

            # Layout
            for pelement in list(pipe_element.elements.values()):
                string = string+"["+pelement.name+"]|\n"
            string = string+"\n"
            for pelement in list(pipe_element.elements.values()):
                string = string+"\n"+self.recursivElement(pelement,parent=parent+"."+pipe_element.name)


        # Pipeline Switch
        elif isinstance(pipe_element, Switch):
            if parent == "":
                string = "[" + pipe_element.name + "]:\n" + "Layout:\n"
            else:
                string = "[" + parent[1:]+"."+pipe_element.name + "]:\n" + "Layout:\n"

            # Layout
            for pelement in pipe_element.pipe_elements:
                string = string + "[" + pelement.name + "]\n"
            string = string + "\n"

            # relate
            string = string + "\n" + "Relate:\n"
            oldElement = ""
            for pelement in pipe_element.pipe_elements:
                if oldElement:
                    string = string + "[" + oldElement + "]" + "<:-:>" + "[" + pelement.name + ']\n'
                oldElement = pelement.name
                string = string + "\n"

            for pelement in pipe_element.pipe_elements:
                string = string + "\n" + self.recursivElement(pelement, parent=parent+"." + pipe_element.name)


        # Pipeline Branch
        elif isinstance(pipe_element, Branch):
            if parent == "":
                string = "[" + pipe_element.name + "]:\n" + "Layout:\n"
            else:
                string = "[" + parent[1:]+"."+pipe_element.name + "]:\n" + "Layout:\n"

            # Layout
            for pelement in pipe_element.pipe_elements:
                string = string + "[" + pelement.name + "]"
            string = string + "\n" + "Relate:\n"
            # Relate
            oldElement = ""
            for pelement in pipe_element.pipe_elements:
                if oldElement:
                    string = string + "[" + oldElement + "]" + "->" + "[" + pelement.name + ']\n'
                oldElement = pelement.name
                string = string + "\n"

            for pelement in pipe_element.pipe_elements:
                string = string + "\n" + self.recursivElement(pelement, parent=parent+"." + pipe_element.name)


        return string


