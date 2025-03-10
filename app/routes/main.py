from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from app.utils.logger import get_logger
from app.services.stock_analyzer import StockAnalyzer
from app.routes.auth import auth_or_login

# Create a blueprint for the main routes
main_bp = Blueprint("main", __name__)

logger = get_logger()


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/tech_analysis")
@auth_or_login
def tech_analysis():
    return render_template("tech_analysis.html")


@main_bp.route("/analyze", methods=["POST"])
@auth_or_login
def analyze():
    try:
        logger.info("开始处理分析请求")
        data = request.json
        stock_codes = data.get("stock_codes", [])
        market_type = data.get("market_type", "A")

        logger.debug(
            f"接收到分析请求: stock_codes={stock_codes}, market_type={market_type}"
        )

        app_config = current_app.config.get("LLM_CONFIGS")
        custom_api_url = app_config.get("API_URL")
        # Get all API keys and rotate through them
        api_keys = app_config.get("API_KEY", "").split(",")

        global last_key_index
        # Initialize last_key_index if it doesn't exist yet
        if "last_key_index" not in globals():
            last_key_index = 0

        # Get the current key
        current_index = last_key_index % len(api_keys)
        custom_api_key = api_keys[current_index].strip()

        # Update index for next use
        last_key_index = (last_key_index + 1) % len(api_keys)

        logger.debug(f"Using API key index {current_index} of {len(api_keys)}")

        custom_api_model = app_config.get("API_MODEL")
        custom_api_timeout = app_config.get("API_TIMEOUT")

        # 创建新的分析器实例，使用自定义配置
        custom_analyzer = StockAnalyzer(
            custom_api_url=custom_api_url,
            custom_api_key=custom_api_key,
            custom_api_model=custom_api_model,
            custom_api_timeout=custom_api_timeout,
        )

        if not stock_codes:
            logger.warning("未提供股票代码")
            return jsonify({"error": "请输入代码"}), 400

        # 使用流式响应
        def generate():
            if len(stock_codes) == 1:
                # 单个股票分析流式处理
                stock_code = stock_codes[0].strip()
                logger.info(f"开始单股流式分析: {stock_code}")

                init_message = (
                    f'{{"stream_type": "single", "stock_code": "{stock_code}"}}\n'
                )
                yield init_message

                logger.debug(f"开始处理股票 {stock_code} 的流式响应")
                chunk_count = 0
                for chunk in custom_analyzer.analyze_stock(
                    stock_code, market_type, stream=True
                ):
                    chunk_count += 1
                    yield chunk + "\n"
                logger.info(
                    f"股票 {stock_code} 流式分析完成，共发送 {chunk_count} 个块"
                )
            else:
                # 批量分析流式处理
                logger.info(f"开始批量流式分析: {stock_codes}")

                init_message = (
                    f'{{"stream_type": "batch", "stock_codes": {stock_codes}}}\n'
                )
                yield init_message

                logger.debug(f"开始处理批量股票的流式响应")
                chunk_count = 0
                for chunk in custom_analyzer.scan_stocks(
                    [code.strip() for code in stock_codes],
                    min_score=0,
                    market_type=market_type,
                    stream=True,
                ):
                    chunk_count += 1
                    yield chunk + "\n"
                logger.info(f"批量流式分析完成，共发送 {chunk_count} 个块")

        logger.info("成功创建流式响应生成器")
        return Response(stream_with_context(generate()), mimetype="application/json")

    except Exception as e:
        error_msg = f"分析时出错: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return jsonify({"error": error_msg}), 500
