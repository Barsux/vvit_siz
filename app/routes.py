import os
import uuid
import time

from app import app, WINDOWS
from flask import render_template, flash, request, redirect, url_for
from threading import Thread
from yolov5.detect_track import run

@app.route("/", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        file_ext = f.filename[f.filename.rfind('.'):]
        new_filename = uuid.uuid1().hex + file_ext
        if WINDOWS:
            new_filename_path = f"{app.config['UPLOADED_PATH']}\\{new_filename}"
        else:
            new_filename_path = f"{app.config['UPLOADED_PATH']}/{new_filename}"
        f.save(os.path.join(app.config['UPLOADED_PATH'], new_filename))
        Thread(target=run, args=(app.config["WEIGHTS_PATH"], new_filename_path, new_filename, False, True)).start()
        return redirect(url_for("show", filename = new_filename, code=307))
    return render_template('index.html')

@app.route("/show/", methods = ["GET", "POST"])
def show():
    return render_template("show.html")