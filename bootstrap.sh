#!/bin/bash

# pip 安装依赖
pip install --no-cache-dir --upgrade -r requirements.txt

# 安装并运行 Gunicorn
pip install gunicorn
gunicorn -w 2 'app:create_app()'