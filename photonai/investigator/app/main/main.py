import os

from flask import Flask


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# base_dir = os.path.join(base_dir, 'investigator')
base_dir = os.path.join(base_dir, 'app')
base_dir = os.path.join(base_dir, 'main')
template_dir = os.path.join(base_dir, 'templates')

application = Flask(__name__, template_folder=template_dir)

application.config['pipe_files'] = {}
application.config['pipe_objects'] = {}

application.config['wizard'] = False

application.config['SECRET_KEY'] = 'Random_Lovely_Key'
application.config['DEBUG'] = True

from pymodm.connection import connect
connect('mongodb://trap-umbriel:27017/photon_results')
# this line is important (add all controllers)
from .controller import default, ajax, hyperpipe, outer_fold, configuration
