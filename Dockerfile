# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \ 
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONPATH=/app

# 复制当前目录下的所有文件到容器的工作目录
COPY . .

# 启动命令
CMD ["bash", "bootstrap.sh"]