from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    request,
    stream_with_context,
)
from app.utils.logger import get_logger
from app.services.stock_analyzer import StockAnalyzer
from app.services.stock_search import StockSearchService
from flask_jwt_extended import jwt_required

# Create a blueprint for the API routes
apis_bp = Blueprint("apis", __name__)

logger = get_logger()


@apis_bp.route("/search", methods=["GET"])
@jwt_required()
def search_stocks():
    """Route for searching stocks by code or name"""
    try:
        keyword = request.args.get("q", "")
        max_results = int(request.args.get("limit", 20))

        if not keyword:
            return jsonify({"error": "请提供搜索关键词"}), 400

        logger.info(f"处理股票搜索请求: 关键词='{keyword}', 最大结果数={max_results}")

        # Initialize the search service
        search_service = StockSearchService()

        # Perform the search
        results = search_service.search_stocks(keyword=keyword, max_results=max_results)

        return jsonify({"success": True, "count": len(results), "results": results})

    except Exception as e:
        error_msg = f"股票搜索出错: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return jsonify({"error": error_msg}), 500


@apis_bp.route("/analyze", methods=["POST"])
@jwt_required()
def analyze():
    try:
        logger.info("开始处理分析请求")
        data = request.json
        stock_code = data.get("stock_code", [])
        market_type = data.get("market_type", "A")

        logger.debug(
            f"接收到分析请求: stock_code={stock_code}, market_type={market_type}"
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

        if not stock_code:
            logger.warning("未提供股票代码")
            return jsonify({"error": "请输入代码"}), 400

        # 使用流式响应
        def generate():
            # 单个股票分析流式处理
            for chunk in custom_analyzer.analyze_stock(stock_code, market_type):
                yield chunk + "\n"

        return Response(stream_with_context(generate()), mimetype="application/json")

    except Exception as e:
        error_msg = f"分析时出错: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return jsonify({"error": error_msg}), 500
