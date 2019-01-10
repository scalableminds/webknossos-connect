# py-datastore
A webKnossos compatible datastore written in Python

## Run it with docker

1. add datastore to wK database `INSERT INTO "webknossos"."datastores"("name","url","key","isscratch","isdeleted","isforeign") VALUES (E'py-datastore',E'http://localhost:8000',E'k',FALSE,FALSE,FALSE);`
2. `./run`

## Development
### Requirements

You need Python 3.7 with pipenv installed. The recommended way is to use pyenv & pipenv:

* Install pyenv via
  `curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash`
* Install your system requirements to build Python, see
  https://github.com/pyenv/pyenv/wiki/common-build-problems
* To install the correct Python version, run
  `pyenv install`
* Start a new shell to activate the Python version:
  `bash`
* Install Pipenv:
  `pip install --user --upgrade pipenv`
* On current Debian & Ubuntu setups, you have to fix a bug manually:
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

### Run it

*  add datastore to wK database `INSERT INTO "webknossos"."datastores"("name","url","key","isscratch","isdeleted","isforeign") VALUES (E'py-datastore',E'http://localhost:8000',E'k',FALSE,FALSE,FALSE);`
* `pipenv install --pre --dev`
* `pipenv run main`
* `curl http://localhost:8000/api/neuroglancer/Connectomics_Department/test -X POST -H "Content-Type: application/json" --data-binary "@neuroglancer.json"`

### Moar

We lint with `pylint` and format with `black`. Use them via

`pipenv run …`
* `pretty`
* `pretty-check`
* `lint`
* `lint-details`


### Development Status

The datastore currently has basic neuroglancer support.

working:
* health endpoint
* pinging wk to report health
* adding neuroglancer datasets
* reporting datasets to wk

not working:
* no proper error handling / status codes (in case of error, exceptions are raised)
* only single channel uint8 data (since wk backend / postgres schema does not support uint64 segmentation)
* does not handle different chunk_sizes well
* currently only considers the first `scale`/`resolution` (since wk client cannot handle all combinations of resolutions)
* thumbnail parameters are hard-coded
* datasets are not persisted
* quite slow, since no caching
* no authentication

housekeeping:
* typing everywhere
* add tests
* linting / code formatting
