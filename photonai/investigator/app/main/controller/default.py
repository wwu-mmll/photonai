from flask import render_template, session
from ..main import application
from .helper import shutdown_server, load_available_pipes


@application.route('/')
def index():
    available_pipes = load_available_pipes()
    return render_template('default/index.html', available_pipes=available_pipes)

@application.route('/investigator_error')
@application.errorhandler(500)
@application.errorhandler(404)
@application.errorhandler(502)
def investigator_error(e):
    error_key = "error_msg"
    success = "Ooopsi, an error occured"
    if e.code == 404:
        success = "...could not find that page."
    elif e.code == 502:
        success = "Probaly MongoDB has some problems. Give it a moment."
    elif error_key in session:
        success = session[error_key]

    session[error_key] = ""
    return render_template('error_page.html', error_msg=success)


@application.route('/shutdown', methods=['GET'])
def shutdown():
    if not application.config['wizard']:
        shutdown_server()
    return render_template('default/shutdown.html')

