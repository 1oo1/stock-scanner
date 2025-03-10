#!/bin/bash

# pip 安装依赖
pip install --no-cache-dir --upgrade -r requirements.txt

pip install gunicorn

# init or upgrade database
flask db upgrade

# run Gunicorn
gunicorn -w 2 'app:create_app()' --bind ${FLASK_HOST}:${FLASK_PORT}