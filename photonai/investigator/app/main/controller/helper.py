from pymodm.connection import connect
from bson.objectid import ObjectId
from pymongo import DESCENDING
from flask import request, session, abort
from pymodm.errors import DoesNotExist, ConnectionError, ValidationError

from photonai.processing.results_structure import MDBHyperpipe, MDBHelper

from ..main import application


def load_pipe_from_db(name):
    try:
        pipe = MDBHyperpipe.objects.order_by([("computation_start_time", DESCENDING)]).raw({'name': name}).first()
        return pipe
    except DoesNotExist as dne:
        # Todo: pretty error handling
        return dne


def load_pipe_from_wizard(obj_id):
    try:
        connect('mongodb://trap-umbriel:27017/photon_results', alias='photon_core')
        pipe = MDBHyperpipe.objects.order_by([("computation_start_time", DESCENDING)]).raw({'wizard_object_id': ObjectId(obj_id)}).first()
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
            connect(application.config['mongo_db_url'], alias='photon_core')
            pipe = load_pipe_from_db(name)

    if storage == "w":
        pipe = load_pipe_from_wizard(name)

    elif storage == "a":
        try:
            pipe = application.config['pipe_objects'][name]
        except KeyError as ke:
            # Todo pretty error handling
            error = ke
    elif storage == "f":
        try:
            pipe_path = application.config['pipe_files'][name]
            pipe = MDBHelper.load_results(pipe_path)
        except KeyError as ke:
            # Todo File not Found
            error= ke
        except Exception as e:
            # Todo: handle file does not exist
            debug = True

    if not pipe:
        session["error_msg"] = "Could not load result object."
        abort(500)
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
    # available_pipes = dict()
    # available_pipes['RAM'] = application.config['pipe_objects'].keys()
    # available_pipes['FILES'] = application.config['pipe_files'].keys()
    # available_pipes['MONGO'] = []
    # if 'mongo_db_url' in application.config:
    #     try:
    #         load_mongo_pipes(available_pipes)
    #     except ValueError as exc:
    #         connect(application.config['mongo_db_url'])
    #         load_mongo_pipes(available_pipes)
    available_pipes = None
    return available_pipes
