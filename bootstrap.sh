#!/bin/bash

# pip 安装依赖
pip install --no-cache-dir --upgrade -r requirements.txt

# init or update database
flask db update

# install and run Gunicorn
pip install gunicorn
gunicorn -w 2 'app:create_app()' --bind ${FLASK_HOST}:${FLASK_PORT}