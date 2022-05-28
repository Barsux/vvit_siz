
# Веб-сервис на основе нейросетевой модели обнаружения с отслеживанием объектов.
## Требования.
Firefox - подобный браузер, chromium браузеры не поддерживаются.
# Установка и запуск.
## Запуск Docker - контейнера с проектом:
```bash
$ git clone https://github.com/Barsux/vvit_siz.git
$ docker build . -t docker_siz
$ docker run -p 0.0.0.0:8080:8080 -p 0.0.0.0:8081:8081 docker_siz
```
## Остановить контейнер можно командой
```bash
$ docker ps 
$ docker stop имя_контейнера
```
# Использование
## Перейдите по ссылке http://127.0.0.1:8080/
<img src="https://github.com/Barsux/vvit_siz/blob/main/examples/src/1.jpg"/>

### Добавтье файл с видео(Несколько готовых примеров находится в директории 'examples/')
### Нажмите кнопку "Upload".
<img src="https://github.com/Barsux/vvit_siz/blob/main/examples/src/2.jpg"/>

### Дождитесь галочки по завершению распознавания и нажмите на кнопку "Go to result"
<img src="https://github.com/Barsux/vvit_siz/blob/main/examples/src/3.jpg"/>

### Наслаждайтесь результатом, также его можно увидеть, перейдя по странице http://127.0.0.1/show .
<img src="https://github.com/Barsux/vvit_siz/blob/main/examples/src/4.jpg"/>