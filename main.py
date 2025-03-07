from src.server import app
from src.utils.logger import get_logger
import os

logger = get_logger()

if __name__ == '__main__':
    flask_env = os.getenv('FLASK_ENV', 'development')
    app.run(host='0.0.0.0', port=8888, debug=(flask_env == 'development'), threaded=True)
    logger.info(f"股票分析系统启动，当前FLASK环境: {flask_env}")