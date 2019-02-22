from flask import render_template
from ..main import application
from .helper import shutdown_server, load_available_pipes


@application.route('/')
def index():
    available_pipes = load_available_pipes()
    return render_template('default/index.html', available_pipes=available_pipes)

@application.route('/shutdown', methods=['GET'])
def shutdown():
    if not application.config['wizard']:
        shutdown_server()
        return render_template('default/shutdown.html')

