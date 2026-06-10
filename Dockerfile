from python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no_cache_dir -r requirements.txt

COPY . .

RUN python main.py

EXPOSE 8000
CMD ["uvicorn","serve:app", "--host","0.0.0.0","--port","8000"]
