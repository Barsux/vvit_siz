from app import app, routes, UPLOAD_FOLDER
from yolov5.detect_track import run
FILENAME_TO_DETECT = "test.mp4"
WEIGHTS = "best.pt"
SHOW_IMG = True

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)

