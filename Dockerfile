FROM python:3.7.2 as prod

RUN pip install poetry==1.0.0b4 && poetry config virtualenvs.create false

RUN mkdir /app
WORKDIR /app

RUN apt-get update && \
    apt-get install -y libturbojpeg0 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY poetry.lock .
RUN poetry install

COPY wkconnect wkconnect
COPY data data
VOLUME /app/data

HEALTHCHECK \
  --interval=15s --timeout=5s --retries=3 \
  CMD curl --fail http://localhost:8000/data/health || exit 1

CMD [ "python", "-m", "wkconnect" ]


FROM prod as dev

RUN poetry install
CMD [ "python", "-m", "wkconnect" ]
