import psycopg2
import requests
import os
import pytest

from app import WEIGHTS_PATH, EXAMPLES_PATH, UPLOAD_PATH, TRACKED_PATH, basedir
from yolov5.db_data import Data_db
from yolov5.detect_track import run as track

TEST_FILENAME = 'zero_file_test.mp4'
HOSTNAME = "http://127.0.0.1:8080"


# Проверка доступности страниц
def test_pages():
    root = requests.get(HOSTNAME)
    show = requests.get(f"{HOSTNAME}/show")
    assert root.status_code == 200 and show.status_code == 200


# Проверка существования и заполненности папкок (Проекта, загрузок, отгрузок, примеров)
def test_dir_root():
    assert os.path.exists(basedir)
    assert len(os.listdir(basedir))


def test_dir_uploads():
    assert os.path.exists(UPLOAD_PATH)
    assert len(os.listdir(UPLOAD_PATH)) == 0


def test_dir_tracked():
    assert os.path.exists(TRACKED_PATH)
    assert len(os.listdir(TRACKED_PATH)) == 0


def test_dir_examples():
    assert os.path.exists(EXAMPLES_PATH)
    assert len(os.listdir(EXAMPLES_PATH))


# Проверка деления на ноль
def test_zerodivision():
    try:
        track(WEIGHTS_PATH, os.path.join(EXAMPLES_PATH, TEST_FILENAME), TEST_FILENAME, False, False)
    except ZeroDivisionError:
        pytest.raises(ZeroDivisionError)


# Проверка существования и доступности бд
def test_db_access():
    a = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="1",
        host="localhost",
        port="5432",
        connect_timeout=5)
    a.close()
    assert a == None


# Проверка существования таблицы для проекта
def test_sql_query():
    sql = ''' select * from information_schema.tables '''
    try:
        a = Data_db()
        cur = a.cursor
        cur.execute(sql)
    except Exception as e:
        pytest.raises(e)
