import os
import uuid
import time

from app import app, WINDOWS
from flask import render_template, flash, request, redirect, url_for
from threading import Thread
from yolov5.detect_track import run
Last_video = None

def run_gst_pipeline(filename):
    print("Running pipeline")
    pipeline = f"gst-launch-1.0 filesrc location=tracked/{filename} ! qtdemux ! decodebin ! videoconvert ! videoscale ! theoraenc ! oggmux ! tcpserversink host=0.0.0.0 port=8081"
    print(pipeline)
    time.sleep(2)
    os.system(pipeline)

@app.route("/", methods = ["GET", "POST"])
def upload():
    global Last_video
    if request.method == 'POST':
        f = request.files.get('file')
        file_ext = f.filename[f.filename.rfind('.'):]
        new_filename = uuid.uuid1().hex + file_ext
        Last_video = new_filename
        if WINDOWS:
            new_filename_path = f"{app.config['UPLOADED_PATH']}\\{new_filename}"
        else:
            new_filename_path = f"{app.config['UPLOADED_PATH']}/{new_filename}"
        f.save(os.path.join(app.config['UPLOADED_PATH'], new_filename))
        detect = Thread(target=run, args=(app.config["WEIGHTS_PATH"], new_filename_path, new_filename, False, False))
        detect.start()
        detect.join()
    return render_template('index.html')

@app.route("/show/", methods = ["GET", "POST"])
def show():
    global Last_video
    if Last_video:
        Thread(target=run_gst_pipeline, args=(Last_video,)).start()
    return render_template("show.html")