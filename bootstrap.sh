#!/bin/bash

# pip 安装依赖
# pip install --no-cache-dir --upgrade -r requirements.txt
pip install --no-cache-dir -r requirements.txt


# 启动服务
# python3 -m uvicorn main:app --host 0.0.0.0 --port 8888
python main.py