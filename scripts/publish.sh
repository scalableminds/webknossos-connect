#!/bin/bash
set -eEuo pipefail

if [ "$#" -ne 1 ]
then
  echo "Usage: ./publish.sh <version>"
  exit 1
fi

PKG_VERSION=$1

echo "Trying to publish $PKG_VERSION"

# Write version to pyproject.toml and version.py
cp pyproject.toml pyproject.toml.bak
sed -i "0,/version = \".*\"/{s/version = \".*\"/version = \"$PKG_VERSION\"/}" pyproject.toml
echo "__version__ = '$PKG_VERSION'" > wkconnect/version.py

poetry publish --build -u $PYPI_USERNAME -p $PYPI_PASSWORD

# Restore files
mv pyproject.toml.bak pyproject.toml
rm wkconnect/version.py
