import os
from sys import platform
from flask_dropzone import Dropzone
from flask import Flask

WINDOWS = (platform == "win32")
WEIGHTS = "best.pt"
UPLOAD_FOLDER = "uploads"
TRACK_FOLDER = "tracked"
if(not os.path.exists(UPLOAD_FOLDER)):
    print("Папка uploads не существовала, я сделалъ")
    os.mkdir(UPLOAD_FOLDER)
else:
    files = os.listdir(UPLOAD_FOLDER)
    if files:
        for file in files:
            os.remove(f"{UPLOAD_FOLDER}\\{file}" if WINDOWS else f"{UPLOAD_FOLDER}/{file}")
        print("Очищена папка upload")
if(not os.path.exists(TRACK_FOLDER)):
    print("Папка tracked не существовала, я сделалъ")
    os.mkdir(TRACK_FOLDER)
else:
    files = os.listdir(TRACK_FOLDER)
    if files:
        for file in files:
            os.remove(f"{TRACK_FOLDER}\\{file}" if WINDOWS else f"{TRACK_FOLDER}/{file}")
        print("Очищена папка tracked")




basedir = os.path.abspath(os.path.dirname(__file__))
if WINDOWS:
    basedir = basedir[:basedir.rfind('\\')]
else:
    basedir = basedir[:basedir.rfind('/')]
print("Корневая директория:",basedir)

app = Flask(__name__)

app.secret_key = b'Y0U_C9NT_BR6TEFORCE_IT'
app.config["UPLOADED_PATH"] = os.path.join(basedir, UPLOAD_FOLDER)
app.config["ROOT_PATH"] = basedir
app.config["WEIGHTS_PATH"] = os.path.join(basedir, WEIGHTS)
app.config["DROPZONE_MAX_FILE_SIZE"]=256
app.config["DROPZONE_TIMEOUT"]=5*60*1000
dropzone = Dropzone(app)