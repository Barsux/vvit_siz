import os
from sys import platform
from flask_dropzone import Dropzone
from flask import Flask

# Константы
DEBUG = False
WEIGHTS = "best.pt"
UPLOAD_FOLDER = "uploads"
TRACK_FOLDER = "tracked"
DEFAULT_WIN_GSTREAMER_PATH = "C:\\GST\\1.0\\msvc_x86_64\\bin"
APP_PORT = 8080
HTTP_PORT = 8081
MAX_FILESIZE = 256  # МБ
RUN_GST_PIPELINE = False if DEBUG else True
EXPORT_DATA_TO_DB = False if DEBUG else True

# Узнаём запущена ли прога на винде
WINDOWS = (platform == "win32")

# Сохраняем корневой путь к папке проекта
basedir = os.path.abspath(os.path.dirname(__file__))
if WINDOWS:
    basedir = basedir[:basedir.rfind('\\')]
else:
    basedir = basedir[:basedir.rfind('/')]
print("Корневая директория:", basedir)
UPLOAD_PATH = os.path.join(basedir, UPLOAD_FOLDER)
TRACKED_PATH = os.path.join(basedir, TRACK_FOLDER)
WEIGHTS_PATH = os.path.join(basedir, WEIGHTS)
EXAMPLES_PATH = os.path.join(basedir, "examples")


# Проверяем на наличие папку uploads, если нет то создаём, если есть - очищаем содержимое.
if not os.path.exists(UPLOAD_FOLDER):
    print("Папка uploads не существовала")
    os.mkdir(UPLOAD_FOLDER)
else:
    files = os.listdir(UPLOAD_FOLDER)
    if files:
        for file in files:
            os.remove(os.path.join(UPLOAD_PATH, file))
        print("Очищена папка upload")
# Проверяем на наличие папку tracked, если нет то создаём, если есть - очищаем содержимое.
if not os.path.exists(TRACK_FOLDER):
    print("Папка tracked не существовала")
    os.mkdir(TRACK_FOLDER)
else:
    files = os.listdir(TRACK_FOLDER)
    if files:
        for file in files:
            os.remove(os.path.join(TRACK_FOLDER, file))
        print("Очищена папка tracked")


# Создаём объект класса Flask
app = Flask(__name__)

# Настраиваем app
app.secret_key = b'Y0U_C9NT_BR6TEFORCE_IT'
app.config["UPLOADED_PATH"] = os.path.join(basedir, UPLOAD_FOLDER)
app.config["ROOT_PATH"] = basedir
app.config["WEIGHTS_PATH"] = WEIGHTS_PATH
app.config["DROPZONE_MAX_FILE_SIZE"] = MAX_FILESIZE
app.config["DROPZONE_TIMEOUT"] = 5 * 60 * 1000
dropzone = Dropzone(app)
