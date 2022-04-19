#Файловый менеджер, скорее всего он будет работать в отдельном потоке, хранить файлы, удалять их
#после истечения срока годности.
import time, queue, os

HOUR = 3600
trackfiles = []
class UploadedFile:
    def __init__(self, filename: str):
        self.filename = filename
        self.detected = False
        self.shelf_life = time.time() + 3600
        self.path = f"./uploaded/{filename}"
        self.tracked_path = ""

    def __del__(self):
        os.remove(self.tracked_path)
        os.remove(self.path)

def is_tracked(filename):
    for trackfile in trackfiles:
        if trackfile.filename == filename and trackfile.detected:
            return True
    return False


def loop(self):
    timestamp = time.time()
    for module_index in len(self.trackfiles):
        file = self.trackfiles[module_index]
        if file.shelf_life <= timestamp:
            self.trackfiles.remove(file)

