FROM restreamio/gstreamer:latest-dev

WORKDIR /app

COPY . /app

RUN apt-get update
RUN apt-get install libavcodec58 ffmpeg libsm6 libxext6 docker-compose -y
RUN pip install -r requirements.txt

EXPOSE 8080 8081 5432 5050

CMD ["docker-compose", "up", "-d", "docker-compose.yaml"]
CMD ["python3", "app.py"]

