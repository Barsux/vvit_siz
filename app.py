from app import routes, APP_PORT, app
#Основной файл программы, запускает её через объект на app
#На порте 8080 на локальном ip
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT)
