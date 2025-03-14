from flask import (
    Blueprint,
    Response,
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
        data = request.json
        stock_code = data.get("stock_code", "")
        stock_name = data.get("stock_name", "")
        market_type = data.get("market_type", "A")

        # 创建新的分析器实例，使用自定义配置
        custom_analyzer = StockAnalyzer()

        if not stock_code:
            logger.warning("未提供股票代码")
            return jsonify({"error": "请输入代码"}), 400

        # 使用流式响应
        def generate():
            # 单个股票分析流式处理
            for chunk in custom_analyzer.analyze_stock(
                stock_code, stock_name, market_type
            ):
                yield chunk + "\n"

        return Response(stream_with_context(generate()), mimetype="application/json")

    except Exception as e:
        error_msg = f"分析时出错: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return jsonify({"error": error_msg}), 500
