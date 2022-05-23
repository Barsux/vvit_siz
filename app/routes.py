import os
import uuid
import time

from app import app
from flask import render_template, flash, request, redirect, url_for
from threading import Thread
from yolov5.detect_track import run
from app.file_mgmt import trackfiles, UploadedFile

@app.route("/", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        file_ext = f.filename[f.filename.rfind('.'):]
        new_filename = uuid.uuid1().hex + file_ext
        new_filename_path = f"{app.config['UPLOADED_PATH']}\\{new_filename}"
        f.save(os.path.join(app.config['UPLOADED_PATH'], new_filename))
        #Thread(target=run, args=(app.config["WEIGHTS_PATH"], new_filename_path)).start()
        return redirect(url_for("show", filename=new_filename))
    return render_template('index.html')

@app.route("/show/<filename>", methods = ["GET", "POST"])
def show(filename):
    return f"Файла {filename} пока нет......\nОтъебитесь!!"

@app.route("/Testroute")
def testrt():
    return redirect(url_for("show", filename="12451.mp4"))