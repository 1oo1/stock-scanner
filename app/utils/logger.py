from loguru import logger
import sys
import os
from datetime import datetime

# 创建日志目录
# Get the project root directory (2 levels up from utils/logger.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)


def init_logger(stdout_level="INFO"):
    """初始化日志配置"""

    # 配置日志
    logger.remove()  # 移除默认的处理器

    # 添加标准输出处理器（控制台）
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=stdout_level,  # Use the determined log level
    )

    # 添加错误日志文件处理器，专门记录错误信息
    logger.add(
        os.path.join(log_dir, "error_{time:YYYY-MM-DD}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} - {message}",
        level="ERROR",
        rotation="00:00",  # 每天午夜轮转
        retention="7 days",  # 保留7天的错误日志
        compression="zip",  # 压缩旧日志文件
        enqueue=True,  # 使用队列写入，提高性能
    )


def clean_old_logs(max_days=7):
    """清理超过指定天数的日志文件"""
    try:
        today = datetime.now()
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            # 跳过目录
            if os.path.isdir(file_path):
                continue

            # 检查文件修改时间
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            days_old = (today - file_time).days

            # 如果文件超过指定天数，删除它
            if days_old > max_days:
                os.remove(file_path)
                logger.info(f"已删除过期日志文件: {filename}")
    except Exception as e:
        logger.error(f"清理日志文件时出错: {e}")


def get_logger():
    """获取通用日志器"""
    # 启动时清理旧日志
    clean_old_logs()
    return logger
