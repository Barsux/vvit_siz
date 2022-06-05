import os
import uuid
import time

from app import app, WINDOWS, DEFAULT_WIN_GSTREAMER_PATH, UPLOAD_PATH, TRACKED_PATH, RUN_GST_PIPELINE, EXPORT_DATA_TO_DB
from flask import render_template, flash, request, redirect, url_for
from threading import Thread
from yolov5.detect_track import run
Last_video = None

#Функция запуска http стрима с помощью gstreamera на порту 8081
def run_gst_pipeline(filename):
    if WINDOWS:
        pipeline = f"{DEFAULT_WIN_GSTREAMER_PATH}\\gst-launch-1.0 filesrc location=tracked/{filename} ! qtdemux ! decodebin ! videoconvert ! videoscale ! theoraenc ! oggmux ! tcpserversink host=0.0.0.0 port=8081"
    else:
        pipeline = f"gst-launch-1.0 filesrc location=tracked/{filename} ! qtdemux ! decodebin ! videoconvert ! videoscale ! theoraenc ! oggmux ! tcpserversink host=0.0.0.0 port=8081"
    os.system(pipeline)


#Корневой маршрут
@app.route("/", methods = ["GET", "POST"])
def upload():
    global Last_video
    if request.method == 'POST':
        #Получает файл из формы
        f = request.files.get('file')
        #Получает полный путь до файла
        filename_path = os.path.join(UPLOAD_PATH, f.filename)
        #Сохраняет файл в папку uploads
        f.save(os.path.join(filename_path))
        #Запускает детектирование, если оно прошло удачно то сохраняет последний файл
        try:
            run(app.config["WEIGHTS_PATH"],  filename_path, f.filename, RUN_GST_PIPELINE, EXPORT_DATA_TO_DB)
        except Exception as e:
            print(e)
        else:
            Last_video = f.filename
    return render_template('index.html')

@app.route("/show/", methods = ["GET", "POST"])
def show():
    global Last_video
    #Если в переменной Last_video чёто есть, то он запускает gstreamerskii - пайплайн в отдельном потоке
    if Last_video:
        Thread(target=run_gst_pipeline, args=(Last_video,)).start()
    return render_template("show.html")