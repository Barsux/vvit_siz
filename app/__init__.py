import os
from sys import platform
from flask_dropzone import Dropzone
from flask import Flask

WINDOWS = (platform == "win32")
WEIGHTS = "best.pt"
UPLOAD_FOLDER = "uploads"
if(not os.path.exists(UPLOAD_FOLDER)):
    print("Папка uploads не существовала, я сделалъ")
    os.mkdir(UPLOAD_FOLDER)

basedir = os.path.abspath(os.path.dirname(__file__))
#Откатил на одну директорию назад, почему-то оно ощущает себя здесь)
print(basedir)
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