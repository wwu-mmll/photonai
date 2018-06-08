from flask import render_template
from photonai.investigator.app.main import app
from photonai.investigator.app.controller.helper import shutdown_server, load_available_pipes


@app.route('/')
def index():
    available_pipes = load_available_pipes()
    return render_template('default/index.html', available_pipes=available_pipes)

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'
