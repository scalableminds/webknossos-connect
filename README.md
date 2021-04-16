# webknossos-connect
A [webKnossos](https://github.com/scalableminds/webknossos) compatible data connector written in Python.

webKnossos-connect serves as an adapter between the webKnossos data store interface and other alternative data storage servers (e.g BossDB) or static files hosted on Cloud Storage (e.g. Neuroglancer Precomputed)

[![Github Actions](https://github.com/scalableminds/webknossos-connect/actions/workflows/main.yml/badge.svg)](https://github.com/scalableminds/webknossos-connect/actions)

Available Adapaters / Supported Data Formats:
- [BossDB](https://bossdb.org/)
- [Neuroglancer Precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)
- [WKW](https://github.com/scalableminds/webknossos-wrap)
- Tiled TIFF

## Usage
### 1. Installation / Docker
Install webKnossos-connect using Docker or use the instructions for native installation below.
`docker-compose up --build webknossos-connect`

### 2. Connecting to webKnossos
Register your webknossos-connect instance with your main webKnossos instance. Modify the webKnossos Postgres database:
  ```
  INSERT INTO "webknossos"."datastores"("name","url","publicurl","key","isscratch","isdeleted","isforeign","isconnector")
  VALUES (E'connect', E'http://localhost:8000', E'http://localhost:8000', E'secret-key', FALSE, FALSE, FALSE, TRUE);
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
    },
    "tiff": {
        "Test Organization": {
            "my_2d_tiff_dataset": {
                "scale": [2.1,2.1]
            }
        }
    }
}
```

Note that tiff datasets are hosted locally. Create compatible tifs with `vips tiffsave source.tif color.tif --tile --pyramid --bigtiff --compression none --tile-width 256 --tile-height 256` and save the generated `color.tif` file at `data/binary/sample_organization/my_2d_tiff_dataset`.

CURL Example
```
curl http:/<webKnossos-connect>/data/datasets -X POST -H "Content-Type: application/json" --data-binary "@datasets.json"
```

#### 3.2 webKnossos UI
Alternatively, new datasets can be added directly through the webKnossos UI. Configure and import a new datasets from the webKnossos dashboard. (Dashboard -> Datasets -> Upload Dataset -> Add wk-connect Dataset)

[Read more in the webKnossos docs.](https://docs.webknossos.org/guides/datasets#uploading-through-the-web-browser)

#### 3.3 Default test datasets

By default, some public datasets are added to webKnossos-connect to get you started when using the Docker image.

## Development
### In Docker :whale:

* Start it with `docker-compose up dev`
* Run other commands `docker-compose run --rm dev pipenv run lint`
* [Check below](#moar) for moar commands.
* If you change the packages, rebuild the image with `docker-compose build dev`

### Native
#### Installation

You need Python 3.8 with `poetry` installed.

```bash
pip install poetry
poetry install
```

#### Run

* Add webknossos-connect to the webKnossos database:
  ```
  INSERT INTO "webknossos"."datastores"("name","url","publicurl","key","isscratch","isdeleted","isforeign","isconnector")
  VALUES (E'connect', E'http://localhost:8000', E'http://localhost:8000', E'secret-key', FALSE, FALSE, FALSE, TRUE);
  ```
* `python -m wkconnect`
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

Use the commands:

* `scripts/pretty.sh`
* `scripts/pretty-check.sh`
* `scripts/lint.sh`
* `scripts/type-check.sh`
* `benchmarks/run_all.sh`

Trace the server on http://localhost:8000/trace.

## License
AGPLv3
Copyright [scalable minds](https://scalableminds.com)
