from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from flask_jwt_extended import jwt_required
from app.utils.logger import get_logger
from app.services.stock_analyzer import StockAnalyzer
from app.routes.auth import auth_or_login

# Create a blueprint for the main routes
main_bp = Blueprint("main", __name__)

logger = get_logger()


@main_bp.route("/")
@auth_or_login
def index():
    logger.info("访问首页")
    return render_template("index.html")


@main_bp.route("/analyze", methods=["POST"])
@jwt_required
def analyze():
    try:
        logger.info("开始处理分析请求")
        data = request.json
        stock_codes = data.get("stock_codes", [])
        market_type = data.get("market_type", "A")

        logger.debug(
            f"接收到分析请求: stock_codes={stock_codes}, market_type={market_type}"
        )

        # 获取自定义API配置
        custom_api_url = data.get("api_url")
        custom_api_key = data.get("api_key")
        custom_api_model = data.get("api_model")
        custom_api_timeout = data.get("api_timeout")

        logger.debug(
            f"自定义API配置: URL={custom_api_url}, 模型={custom_api_model}, API Key={'已提供' if custom_api_key else '未提供'}, Timeout={custom_api_timeout}"
        )

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
