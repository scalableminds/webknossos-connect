FROM python:3.8.8 as prod

RUN pip install poetry==1.1.6

RUN mkdir /app
WORKDIR /app

RUN apt-get update && \
    apt-get install -y libturbojpeg0 liblz4-dev curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl https://sh.rustup.rs -sSf | bash -s -- -y

COPY pyproject.toml .
COPY poetry.lock .
COPY poetry.toml .
RUN poetry install

# Run install again for installing wkconnect globally
COPY wkconnect wkconnect
RUN poetry install

COPY fast_wkw fast_wkw
RUN curl --output wkw.zip https://codeload.github.com/scalableminds/webknossos-wrap/zip/refs/heads/lz4-crate && \
    unzip wkw.zip && \
    cd fast_wkw && \
    source $HOME/.cargo/env && \
    cargo build --release && \
    cp target/release/libfast_wkw*.so ../wkconnect/fast_wkw.so && \
    cd .. && \
    rm -r fast_wkw

COPY data data
VOLUME /app/data

HEALTHCHECK \
  --interval=15s --timeout=5s --retries=3 \
  CMD curl --fail http://localhost:8000/data/health || exit 1

CMD [ "python", "-m", "wkconnect" ]


FROM prod as dev

RUN poetry install
CMD [ "python", "-m", "wkconnect" ]
