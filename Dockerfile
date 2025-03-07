# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ca-certificates \ 
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
# ENV PYTHONPATH=/app

# 暴露端口（如果需要）
EXPOSE 8888

# 启动命令
CMD ["bash", "bootstrap.sh"]