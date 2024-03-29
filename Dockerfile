FROM python:3.8.8 as prod

RUN pip install poetry==1.1.6

RUN mkdir /app
WORKDIR /app

RUN apt-get update && \
    apt-get install -y libturbojpeg0 liblz4-dev curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y

COPY pyproject.toml .
COPY poetry.lock .
COPY poetry.toml .
COPY build.py .
COPY README.md .
COPY fast_wkw fast_wkw
COPY wkconnect wkconnect

RUN PATH="$HOME/.cargo/bin:${PATH}" poetry install

COPY data data
VOLUME /app/data

HEALTHCHECK \
  --interval=15s --timeout=5s --retries=3 \
  CMD curl --fail http://localhost:8000/data/health || exit 1

CMD [ "python", "-m", "wkconnect" ]


FROM prod as dev

RUN poetry install
CMD [ "python", "-m", "wkconnect" ]
