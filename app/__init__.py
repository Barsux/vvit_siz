import os
from flask import Flask



app = Flask(__name__)
app.secret_key = b'Y0U_C9NT_BR6TEFORCE_IT'
app.config["UPLOAD_FOLDER"] = './uploaded'
