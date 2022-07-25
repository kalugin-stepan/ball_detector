# Обнаружение мяча

Это библиотека для Python, которая позволяет обнаружить мяч на изображении.

## Зависимости

* OpenCv 4.3

## Сборка

Для начала в файле `setup.py` надо поуказать путь до заголовочных файлов и библиотек opencv.
Перед компиляцией в файле `main.cpp` надо указать `cascadePath` - путь до файла `ball_cascade.xml`.

### Для windows

```bash
python setup.py build
```

После чего скомпилированную библиотеку из папки `build` надо перенести в Python\Python3x\DLLs, туда же надо перенести и dll файлы из opencv.

### Для linux

```bash
python3 setup.py install
```
