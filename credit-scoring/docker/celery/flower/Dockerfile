FROM python:3.12.4-slim

WORKDIR /app

COPY ./docker/celery/flower/requirements.txt /tmp
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

COPY ./src /app/src
