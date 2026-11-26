# Берем легкую версию Python
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Dockerfile
# ... (начало файла без изменений) ...

COPY requirements.txt .

# Используем зеркало depkit.ru
RUN pip install --no-cache-dir --default-timeout=1000 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt


# Копируем весь остальной код проекта
COPY . .

# Открываем порт
EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]