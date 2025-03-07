import os
import requests
import json
from utils.logger import get_logger, get_stream_logger
from dotenv import load_dotenv
from utils.api_utils import APIUtils

# 获取日志器
logger = get_logger()
stream_logger = get_stream_logger()

def _truncate_json_for_logging(json_obj, max_length=500):
    """截断JSON对象用于日志记录，避免日志过大
    
    Args:
        json_obj: 要截断的JSON对象
        max_length: 最大字符长度，默认500
        
    Returns:
        str: 截断后的JSON字符串
    """
    if isinstance(json_obj, str):
        json_str = json_obj
    else:
        json_str = json.dumps(json_obj, ensure_ascii=False)
    
    if len(json_str) <= max_length:
        return json_str
    return json_str[:max_length] + f"... [截断，总长度: {len(json_str)}字符]"

def test_api_stream():
    """
    测试API流式响应功能
    """
    # 加载环境变量
    load_dotenv()
    
    # 获取API配置
    api_url = os.getenv('API_URL')
    api_key = os.getenv('API_KEY')
    api_model = os.getenv('API_MODEL', 'gpt-3.5-turbo')
    
    logger.info(f"开始测试API流式响应，API URL: {api_url}, MODEL: {api_model}")
    
    # 检查API配置
    if not api_url:
        logger.error("API URL未配置，无法进行测试")
        return
        
    if not api_key:
        logger.error("API Key未配置，无法进行测试")
        return
    
    # 标准化API URL
    api_url = APIUtils.format_api_url(api_url)
    logger.debug(f"标准化后的API URL: {api_url}")
    
    # 构建简单的测试提示
    prompt = "这是一个API流式响应测试。请给出一个简短的股票分析样例。"
    
    # 构建请求头和请求体
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True  # 明确设置stream参数为True
    }
    
    logger.debug(f"请求载荷: {_truncate_json_for_logging(payload)}")
    
    try:
        logger.info(f"发起流式API请求: {api_url}")
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=int(os.getenv('API_TIMEOUT', 60)),
            stream=True
        )
        
        logger.info(f"API流式响应状态码: {response.status_code}")
        logger.debug(f"响应头: {response.headers}")
        
        if response.status_code == 200:
            logger.info("成功获取API流式响应，开始处理")
            
            buffer = ""
            chunk_count = 0
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    logger.info(f"原始流式行: {line_str}")
                    
                    # 跳过保持连接的空行
                    if line_str.strip() == '':
                        logger.debug("跳过空行")
                        continue
                    
                    # 数据行通常以"data: "开头
                    if line_str.startswith('data: '):
                        data_content = line_str[6:].strip()  # 移除 "data: " 前缀并去除前后空格
                        logger.info(f"数据内容: {data_content}")
                        
                        # 检查是否为流的结束
                        if data_content == '[DONE]':
                            logger.info("收到流结束标记 [DONE]")
                            break
                            
                        try:
                            # 解析JSON数据
                            json_data = json.loads(data_content)
                            logger.debug(f"JSON结构: {_truncate_json_for_logging(json_data)}")
                            
                            if 'choices' in json_data:
                                delta = json_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    chunk_count += 1
                                    buffer += content
                                    logger.info(f"内容片段 #{chunk_count}: {content}")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误: {e}, 内容: {data_content}")
                    else:
                        logger.warning(f"收到非'data:'开头的行: {line_str}")
            
            logger.info(f"流式处理完成，共收到 {chunk_count} 个内容片段")
            logger.info(f"完整内容:\n{buffer}")
            
        else:
            try:
                error_response = response.json()
                error_text = json.dumps(error_response, indent=2)
            except:
                error_text = response.text[:500] if response.text else "无响应内容"
                
            logger.error(f"API请求失败: 状态码 {response.status_code}, 响应: {error_text}")
            
    except Exception as e:
        logger.error(f"测试过程中发生异常: {str(e)}")
        logger.exception(e)

if __name__ == "__main__":
    test_api_stream()
