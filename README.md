# Кровавая техника безопасности)

Что сделано?
Да особо ничего)

## Что не реализовано:
```
- Вывод потокового видео с помощью Gstreamer
- Сохранение данных детектора в БД
- Настройка контейнера Docker и развёртывания программы в ней
```
## Что реализовано:
```
- Обучена нейросетка, получены веса
- Веб морда
- Бэк (Только залив файла на сервер)
- Детектор
```



## Чтобы запустить этот огрызок:
#### Требуется >= Python3.9

### Клонируем репозиторй по HTTP(s)/SSH
```
git clone https://github.com/Barsux/vvit_siz.git
```
### Устанавливаем зависимости:

### На Windows:
```powershell
- python -m venv env
- env\scripts\activate.bat
- pip install -r requirements.txt
```

### На Linux:
```bash
- $ python3 -m venv env
- $ source env/bin/activate
- $ pip3 install requirements.txt
```

## Запуск программы пока выполняется командой:

### Windows:
```
- python app.py
```
### Linux:

```
- $ python3 app.py
```