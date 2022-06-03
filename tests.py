import requests
URL = "http://barsux.tech:8080"
ROOT_ROUTE = '/'
SHOW_ROUTE = '/show'
def test_root_route():
    result = requests.get(f"{URL}/{ROOT_ROUTE}")
    assert result.status_code == 200

def test_show_route():
    result = requests.get(f"{URL}/{SHOW_ROUTE}")
    assert result.status_code == 200