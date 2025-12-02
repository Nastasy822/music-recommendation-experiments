FROM python:3.11.9-slim AS builder

ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt
#
RUN python -m pip install --upgrade pip \
    && apt-get update \
    && apt-get -y install gcc python3-dev \
    && apt-get install -y --no-install-recommends git
RUN apt-get install unzip
#
#
RUN python -m venv venv
RUN pip install -U pip setuptools
RUN pip install poetry==1.2.2
#
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --no-root
RUN poetry shell

RUN pip install faiss-cpu==1.13.0

COPY . .