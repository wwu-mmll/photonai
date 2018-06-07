from flask import render_template
from photonai.investigator.app.main import app
from photonai.investigator.app.controller.helper import shutdown_server


@app.route('/')
def index():
    return render_template('default/index.html')

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'
