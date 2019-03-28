# webknossos-connect
A [webKnossos](https://github.com/scalableminds/webknossos) compatible data connector written in Python. 

webKnossos-connect serves as an adapter between the webKnossos data store interface and other alternative data storage servers (e.g BossDB) or static files hosted on Google Cloud Storage (e.g. Neurodata Precomputed)

[![CircleCI](https://circleci.com/gh/scalableminds/webknossos-connect.svg?style=svg&circle-token=1d7b55b40a5733c7563033064cee0ed0beef36b6)](https://circleci.com/gh/scalableminds/webknossos-connect)

Available Adapaters / Supported Data Formats:
- [BossDB](https://bossdb.org/)
- [Neuroglancer Precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)

## Usage
### 1. Installation / Docker
Install webKnossos-connect using Docker or use the instructions for native installation below.
`docker-compose up --build webknossos-connect`

### 2. Connecting to webKnossos
Register your webknossos-connect instance with your main webKnossos instance. Modify the webKnossos Postgres database:
  ```
  INSERT INTO "webknossos"."datastores"("name","url","key","isscratch","isdeleted","isforeign","isconnector")
  VALUES (E'connect', E'http://localhost:8000', E'secret-key', FALSE, FALSE, FALSE, TRUE);
  ```
### 3. Adding Datasets
Add and configure datasets to webKnossos-connect to make them available for viewing in webKnossos

#### 3.1 REST API
You can add new datasets to webKnossos-connect through the REST interface. POST a JSON configuration to:
```
http://<webKnossos-connect>/data/datasets?token
```
The access `token` can be obained from your user profile in the webKnossos main instance. [Read more in the webKnosssos docs.](https://docs.webknossos.org/reference/rest_api#authentication)

Example JSON body. More examples can be found [here](https://github.com/scalableminds/webknossos-connect/blob/master/data/datasets.json).
```
{
    "boss": {
        "Test Organisation": {
            "ara": {
                "domain": "https://api.boss.neurodata.io",
                "collection": "ara_2016",
                "experiment": "sagittal_50um",
                "username": "<NEURODATA_IO_USER>",
                "password": "<NEURODATA_IO_PW>"
            },
        }
    },
    "neuroglancer": {
        "Test Organisation": {
            "fafb_v14": {
                "layers": {
                    "image": {
                        "source": "gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe",
                        "type": "image"
                    }
                }
            }
        }
    }
}
```

CURL Example
```
curl http:/<webKnossos-connect>/data/datasets -X POST -H "Content-Type: application/json" --data-binary "@datasets.json"
```

#### 3.2 webKnossos UI
Alternatively, new datasets can be added directly through the webKnossos UI. Configure and import a new datasets from the webKnossos dashboard. (Dashboard -> Datasets -> Upload Dataset -> Add wk-connect Dataset) 

(Read more in the webKnossos docs.)[https://docs.webknossos.org/guides/datasets#uploading-through-the-web-browser]

#### 3.2 Default test datasets

By default, some public datasets are added to webKnossos-connect to get you started when using the Docker image. Some initial datasets are hosted on [neurodata.io](https://neurodata.io/ndcloud/#data). For access, create a `.env` file with your Neurodata.io credentials:
  ```
  NEURODATA_IO_USER="<your username>"
  NEURODATA_IO_PW="<your password>"
  ```


## Development
### In Docker :whale:

* Start it with `docker-compose up dev`
* Run other commands `docker-compose run --rm dev pipenv run lint`
* [Check below](#moar) for moar commands.
* If you change the packages, rebuild the image with `docker-compose build dev`

### Native
#### Installation

You need Python 3.7 with `pipenv` installed.
The recommended way is to use `pyenv` and `pipenv`:

* Install `pyenv` with  
  `curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash`
* Install your system requirements to build Python, see
  https://github.com/pyenv/pyenv/wiki/common-build-problems
* To install the correct Python version, run
  `pyenv install`
* Start a new shell to activate the Python version:
  `bash`
* Install Pipenv:
  `pip install --user --upgrade pipenv`
* On current Debian and Ubuntu setups, you have to fix a bug manually:
  Add `~/.local/bin` to your PATH like this:
  ```
  echo '
  # set PATH so it includes the private local bin if it exists
  # This should be the default, but is broken in some Debian/Ubuntu Versions,
  # see https://unix.stackexchange.com/questions/316765/which-distributions-have-home-local-bin-in-path#answer-392710
  if [ -d "$HOME/.local/bin" ] ; then
      PATH="$HOME/.local/bin:$PATH"
  fi
  ' >> ~/.profile
  ```
  This will be activated after the next login automatically, to use it right now, run
  `. ~/.profile`

#### Run

* Add webknossos-connect to the webKnossos database:
  ```
  INSERT INTO "webknossos"."datastores"("name","url","key","isscratch","isdeleted","isforeign") 
  VALUES (E'connect',E'http://localhost:8000',E'k',FALSE,FALSE,FALSE);
  ```
* `pipenv install --dev`
* `pipenv run main`
* ```
  curl http://localhost:8000/api/neuroglancer/Demo_Lab/test \
    -X POST -H "Content-Type: application/json" \
    --data-binary "@datasets.json"
  ```

### Moar

Useful commands:

* Lint with `pylint` & `flake8`
* Format with `black`, `isort` & `autoflake`
* Type-check with `mypy`
* Benchark with `timeit`
* Trace with `py-spy`

Use the commands with `pipenv run …`:

* `pretty`
* `pretty-check`
* `lint`
* `lint-details`
* `type-check`
* `benchmarks/run_all.sh`

Trace the server on http://localhost:8000/trace.

## License
AGPLv3
Copyright [scalable minds](https://scalableminds.com)
