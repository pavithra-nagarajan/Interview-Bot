FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "/app/ai-analysis.py"]
