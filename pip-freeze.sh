#! /bin/sh
cd `dirname $0`
.venv/bin/pip freeze > requirements.txt

