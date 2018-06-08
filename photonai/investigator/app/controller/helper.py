from photonai.validation.ResultsDatabase import MDBHyperpipe
from pymodm.errors import DoesNotExist, ConnectionError, ValidationError
from photonai.investigator.app.main import app
from photonai.validation.ResultsDatabase import MDBHelper
from pymodm.connection import connect
from flask import request, redirect, url_for


def load_pipe_from_db(name):
    try:
        pipe = MDBHyperpipe.objects.get({'_id': name})
        return pipe
    except DoesNotExist as dne:
        # Todo: pretty error handling
        return dne


def load_pipe(storage, name):
    pipe = None
    error = "Could not load pipeline"
    if storage == "m":
        try:
            pipe = load_pipe_from_db(name)
        except ValueError as exc:
            connect(app.config['mongo_db_url'])
            pipe = load_pipe_from_db(name)
    elif storage == "a":
        try:
            pipe = app.config['pipe_objects'][name]
        except KeyError as ke:
            # Todo pretty error handling
            error = ke
    elif storage == "f":
        try:
            pipe_path = app.config['pipe_files'][name]
            pipe = MDBHelper.load_results(pipe_path)
        except KeyError as ke:
            # Todo File not Found
            error= ke
        except Exception as e:
            # Todo: handle file does not exist
            debug = True

    if not pipe or not isinstance(pipe, MDBHyperpipe):
        # return "Could not load pipeline"
        return error
    return pipe


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


def load_mongo_pipes(available_pipes):
    try:
        # Todo: load only name
        pipeline_list = list(MDBHyperpipe.objects.all())
        for item in pipeline_list:
            available_pipes['MONGO'].append(item.pk)
    except ValidationError as exc:
        return exc
    except ConnectionError as exc:
        return exc


def load_available_pipes():
    available_pipes = dict()
    available_pipes['RAM'] = app.config['pipe_objects'].keys()
    available_pipes['FILES'] = app.config['pipe_files'].keys()
    available_pipes['MONGO'] = []
    if 'mongo_db_url' in app.config:
        try:
            load_mongo_pipes(available_pipes)
        except ValueError as exc:
            connect(app.config['mongo_db_url'])
            load_mongo_pipes(available_pipes)
    return available_pipes
