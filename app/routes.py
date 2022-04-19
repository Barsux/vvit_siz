import os
from app import app
from flask import render_template, flash, request
from file_mgmt import trackfiles, UploadedFile, is_tracked

@app.route("/", methods = ["GET", "POST"])
def rootroute():
    return render_template("root.html")

@app.route("/track", methods = ["GET", "POST"])
def track():
    if request.method == 'POST':
        f = request.files['upfile']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        trackfiles.append(UploadedFile(f.filename))
        while not is_tracked(f.filename):
            pass
        #Как-то нужно вернуть эту хуйню назад
    return render_template("track.html")