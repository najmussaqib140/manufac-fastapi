#install python
#install docker
FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy your code
COPY app /app/app

EXPOSE 8000

# Start FastAPI
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
