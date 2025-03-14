# 使用 Python 3.12 作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \ 
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PYTHONPATH=/app

# 复制文件。忽略的文件在 .dockerignore 中定义
COPY . .

# 设置执行权限（在切换用户之前）
RUN chmod +x bootstrap.sh

# 启动命令
CMD ["./bootstrap.sh"]