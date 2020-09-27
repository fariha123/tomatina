# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
set FLASK_APP=abc.py
python -m flask run
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!


if __name__ == '__main__':
    app.run(debug = True)
