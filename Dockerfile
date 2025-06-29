FROM tensorflow/tensorflow:2.19.0-gpu-jupyter

WORKDIR /app

COPY . /app

RUN #pip install --cache-dir=/tmp/pip_cache -r requirements.txt

CMD ["python3", "/app/tensorflowlearn/classification.py"]

EXPOSE 8888