# Используем официальный образ Python slim
FROM python:3.10.11-slim

# Устанавливаем Git, sentencepiece и accelerate
RUN apt-get update && apt-get install -y git && pip install --no-cache-dir sentencepiece accelerate

# Устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

# Экспортируем порт 5000
EXPOSE 5000

# Запускаем приложение
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]