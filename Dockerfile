FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


EXPOSE 8080

CMD ["python3", "app.py"]

