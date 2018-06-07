import os

from flask import Flask

from pymodm.connection import  connect

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
base_dir = os.path.join(base_dir, 'investigator')
base_dir = os.path.join(base_dir, 'app')
template_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

app.config['SECRET_KEY'] = 'Random_Lovely_Key'
app.config['DEBUG'] = False
connect("mongodb://localhost:27017/photon_db")

# this line is important (add all controllers)
from photonai.investigator.app.controller import default, hyperpipe, outer_fold, configuration, ajax
