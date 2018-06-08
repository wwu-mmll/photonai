from photonai.validation.ResultsDatabase import MDBHyperpipe
from pymodm.errors import DoesNotExist, ConnectionError
from photonai.investigator.app.main import app
from pymodm.connection import connect
from flask import request, render_template

def load_pipe_from_db(name):
    try:
        pipe = MDBHyperpipe.objects.get({'_id': name})
        return pipe
    except DoesNotExist as dne:
        # Todo: pretty error handling
        return dne


def load_pipe(storage, name):
    pipe = None
    if storage == "m":
        try:
            pipe = load_pipe_from_db(name)
        except ConnectionError as exc:
            # if we are not connected yet, do it
            connect(app.config['mongo_db_url'])
            pipe = load_pipe_from_db(name)
        except ValueError as exc:
            connect(app.config['mongo_db_url'])
            pipe = load_pipe_from_db(name)
    elif storage == "a":
        try:
            pipe = app.config['pipe_objects'][name]
        except KeyError as ke:
            # Todo pretty error handling
            return ke

    if not pipe:
        return render_template('error.html', error_msg="Could not load pipeline")
    return pipe


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
