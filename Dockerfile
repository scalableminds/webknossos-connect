FROM python:3

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY py-datastore/ /app/py-datastore

ENTRYPOINT [ "python", "-m", "py-datastore" ]
