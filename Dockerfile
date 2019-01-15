FROM python:3.7.2 as prod

RUN pip install --user --upgrade pipenv
# We have to set PATH by hand because of this bug:
# https://unix.stackexchange.com/questions/316765/which-distributions-have-home-local-bin-in-path#answer-392710
ENV PATH=/root/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN mkdir /app
WORKDIR /app

COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --system

COPY py_datastore py_datastore

CMD [ "python", "-m", "py_datastore" ]


FROM prod as dev

RUN pipenv sync --dev
CMD [ "pipenv", "run", "main" ]
