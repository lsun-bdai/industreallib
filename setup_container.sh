#! /bin/bash
# this script is used to setup the container after it is started

# run pip install -e . in the file directory to install industreallib and its dependencies
cd "$(dirname "$0")"
pip install -e .

# install numpy version 1.22 to avoid float type issues
pip install numpy==1.22
