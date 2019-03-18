FROM python:3.7.2 as prod

RUN pip install --user --upgrade pipenv
# We have to set PATH by hand because of this bug:
# https://unix.stackexchange.com/questions/316765/which-distributions-have-home-local-bin-in-path#answer-392710
ENV PATH=/root/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN mkdir /app
WORKDIR /app

RUN apt-get update && \
    apt-get install -y libturbojpeg0 && \
    rm -rf /var/lib/apt/lists/*

COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --system

COPY wkconnect wkconnect

HEALTHCHECK \
  --interval=15s --timeout=5s --retries=3 \
  CMD curl --fail http://localhost:8000/data/health || exit 1

CMD [ "python", "-m", "wkconnect" ]


FROM prod as dev

RUN pipenv sync --dev
CMD [ "pipenv", "run", "main" ]
