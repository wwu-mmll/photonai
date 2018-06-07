from flask import render_template
from photonai.investigator.app.main import app


@app.route('/')
def index():
    return render_template('default/index.html')
