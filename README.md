# webknossos-connect
A [webKnossos](https://github.com/scalableminds/webknossos) compatible data connector written in Python

[![CircleCI](https://circleci.com/gh/scalableminds/webknossos-connect.svg?style=svg&circle-token=1d7b55b40a5733c7563033064cee0ed0beef36b6)](https://circleci.com/gh/scalableminds/webknossos-connect)

## Usage

1. Add webknossos-connect to the webKnossos database:
  ```
  INSERT INTO "webknossos"."datastores"("name","url","key","isscratch","isdeleted","isforeign") 
  VALUES (E'connect',E'http://localhost:8000',E'k',FALSE,FALSE,FALSE);
  ```
2. `docker-compose up --build webknossos-connect`
3. By default, some public datasets are reported. Add more datasets from the webKnossos user interface.

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
