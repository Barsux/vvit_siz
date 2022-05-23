import os
from app import app
from flask import render_template, flash, request
from app.file_mgmt import trackfiles, UploadedFile

@app.route("/", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    return render_template('index.html')
