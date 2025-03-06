import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
from typing import Dict, List, Optional, Tuple, Generator
from dotenv import load_dotenv
import json
from logger import get_logger
from utils.api_utils import APIUtils

# 获取日志器
logger = get_logger()

class StockAnalyzer:
    def __init__(self, initial_cash=1000000, custom_api_url=None, custom_api_key=None, custom_api_model=None, custom_api_timeout=None):
        
        # 加载环境变量
        load_dotenv()
        
        # 设置 API 配置，优先使用自定义配置，否则使用环境变量
        self.API_URL = custom_api_url or os.getenv('API_URL')
        self.API_KEY = custom_api_key or os.getenv('API_KEY')
        self.API_MODEL = custom_api_model or os.getenv('API_MODEL', 'gpt-3.5-turbo')
        self.API_TIMEOUT = int(custom_api_timeout or os.getenv('API_TIMEOUT', 60))
        
        logger.debug(f"初始化StockAnalyzer: API_URL={self.API_URL}, API_MODEL={self.API_MODEL}, API_KEY={'已提供' if self.API_KEY else '未提供'}, API_TIMEOUT={self.API_TIMEOUT}")
        
        # 配置参数 - 分市场类型设置不同参数
        self.market_params = {
            'A': {
                'ma_periods': {'short': 5, 'medium': 20, 'long': 60},
                'rsi_period': 14,
                'bollinger_period': 20,
                'bollinger_std': 2,
                'volume_ma_period': 20,
                'atr_period': 14,
                'volume_ratio_threshold': {'normal': 1.5, 'extreme': 3.0},
                'price_change_threshold': 0.03
            },
            'HK': {
                'ma_periods': {'short': 5, 'medium': 20, 'long': 60},
                'rsi_period': 20,  # 港股适合更长周期RSI
                'bollinger_period': 20,
                'bollinger_std': 2,
                'volume_ma_period': 20,
                'atr_period': 14,
                'volume_ratio_threshold': {'normal': 1.2, 'extreme': 2.5},
                'price_change_threshold': 0.02
            }
        }
        
        # 默认使用A股参数
        self.params = self.market_params['A']

        
    def get_stock_data(self, stock_code, market_type='A', start_date=None, end_date=None, ):
        """获取股票数据"""
        import akshare as ak
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        try:
            # 根据市场类型获取数据
            if market_type == 'A':
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                # A股数据列名映射
            elif market_type == 'HK':
                df = ak.stock_hk_daily(
                    symbol=stock_code,
                    adjust="qfq"
                )
            
            else:
                raise ValueError(f"不支持的市场类型: {market_type}")
            
            # 重命名列名以匹配分析需求
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume"
            })
            
            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])
            
            # 数据类型转换
            numeric_columns = ['open', 'close', 'high', 'low', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # 删除空值
            df = df.dropna()
            
            return df.sort_values('date')
            
        except Exception as e:
            raise Exception(f"获取股票数据失败: {str(e)}")
            
    def calculate_ema(self, series, period, method='ema'):
        """计算指数移动平均线或简单移动平均线"""

        # 根据方法选择不同的均线类型
        if method == 'sma' or period == self.params['ma_periods']['long']:
            return series.rolling(window=period).mean()
        else:
            # 使用adjust=True修正EMA前期权重偏差
            return series.ewm(span=period, adjust=True).mean()
        
    def calculate_rsi(self, series, period):
        """计算RSI指标 - 使用Wilder's方法"""
        delta = series.diff()
        # 使用Wilder's smoothing
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        down = abs(down)
        
        # 安全检查数据长度
        if len(series) <= period:
            logger.warning(f"数据长度({len(series)})不足以计算{period}日RSI")
            return pd.Series(np.nan, index=series.index)
            
        # 修正：使用前period日数据初始化平均值，符合标准Wilder方法
        try:
            avg_gain = up[:period].mean()
            avg_loss = down[:period].mean()
        except Exception as e:
            logger.error(f"RSI初始平均值计算错误: {str(e)}")
            return pd.Series(np.nan, index=series.index)
        
        # 使用安全的方式计算后续值
        gains = pd.Series(np.nan, index=series.index)
        losses = pd.Series(np.nan, index=series.index)
        
        # 设置初始值
        gains.iloc[period] = avg_gain
        losses.iloc[period] = avg_loss
        
        # 计算后续值
        for i in range(period+1, len(series)):
            try:
                gains.iloc[i] = (gains.iloc[i-1] * (period-1) + up.iloc[i]) / period
                losses.iloc[i] = (losses.iloc[i-1] * (period-1) + down.iloc[i]) / period
            except Exception as e:
                logger.error(f"RSI计算第{i}个点时出错: {str(e)}")
                # 保持为NaN
                
        # 避免除以0，并处理NaN
        rs = gains / losses.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_macd(self, series):
        """计算MACD指标"""
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
        
    def calculate_bollinger_bands(self, series, period, std_dev):
        """计算布林带"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # 计算带宽收窄检测
        band_width = (upper - lower) / middle * 100
        band_width_ma = band_width.rolling(window=20).mean()
        band_width_narrowing = band_width < (0.85 * band_width_ma)  # 带宽小于20日带宽均值的85%视为收窄
        
        return upper, middle, lower, band_width_narrowing
        
    def calculate_atr(self, df, period, method='wilder'):
        """计算ATR指标 - 支持不同平滑方法"""
        if len(df) <= period:
            logger.warning(f"数据长度({len(df)})不足以计算{period}日ATR")
            return pd.Series(np.nan, index=df.index)
            
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        # 处理第一个点的前一日收盘价缺失问题
        close.iloc[0] = df['open'].iloc[0]  # 使用开盘价替代
        
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 特殊处理A股涨跌停情况
        # 当日最高价=当日最低价可能是涨停或跌停
        possible_limit = (high == low)
        
        # 区分科创板/创业板与主板不同涨跌幅限制
        def is_star_chinext(code):
            # 简单判断：科创板(688开头)或创业板(300开头)
            if isinstance(code, str):
                return code.startswith('688') or code.startswith('300')
            return False
            
        # 获取股票代码 (如果代码作为属性存在)
        stock_code = getattr(self, 'current_stock_code', '')
            
        # 对可能的涨跌停日，修正TR值
        if is_star_chinext(stock_code):
            # 科创板/创业板20%涨跌幅
            tr[possible_limit] = df.loc[possible_limit, 'close'] * 0.05  # 假设5%的波动
        else:
            # 主板10%涨跌幅
            tr[possible_limit] = df.loc[possible_limit, 'close'] * 0.02  # 假设2%的波动
        
        # 安全计算ATR
        atr = pd.Series(np.nan, index=tr.index)
        
        # 首个值使用简单平均
        if period < len(tr):
            atr.iloc[period] = tr.iloc[:period+1].mean()
            
            # 根据不同平滑方法计算后续值
            if method == 'wilder':
                # Wilder的平滑方法
                for i in range(period+1, len(tr)):
                    try:
                        atr.iloc[i] = (atr.iloc[i-1] * (period-1) + tr.iloc[i]) / period
                    except Exception as e:
                        logger.error(f"ATR(Wilder)计算第{i}个点时出错: {str(e)}")
            elif method == 'sma':
                # 简单移动平均
                atr.iloc[period+1:] = tr.iloc[period+1:].rolling(window=period).mean()
            elif method == 'ema':
                # 指数移动平均
                for i in range(period+1, len(tr)):
                    try:
                        atr.iloc[i] = atr.iloc[i-1] * (1 - 2/(period+1)) + tr.iloc[i] * (2/(period+1))
                    except Exception as e:
                        logger.error(f"ATR(EMA)计算第{i}个点时出错: {str(e)}")
        
        return atr
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        try:
            # 计算移动平均线
            df['MA5'] = self.calculate_ema(df['close'], self.params['ma_periods']['short'])
            df['MA20'] = self.calculate_ema(df['close'], self.params['ma_periods']['medium'])
            df['MA60'] = self.calculate_ema(df['close'], self.params['ma_periods']['long'], method='sma')  # 长期均线使用SMA
            
            # 计算RSI
            df['RSI'] = self.calculate_rsi(df['close'], self.params['rsi_period'])
            
            # 计算MACD
            df['MACD'], df['Signal'], df['MACD_hist'] = self.calculate_macd(df['close'])
            
            # 计算布林带
            df['BB_upper'], df['BB_middle'], df['BB_lower'], df['BB_narrowing'] = self.calculate_bollinger_bands(
                df['close'],
                self.params['bollinger_period'],
                self.params['bollinger_std']
            )
            
            # 成交量分析
            df['Volume_MA'] = df['volume'].rolling(window=self.params['volume_ma_period']).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
            
            # 资金流向指标(MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = pd.Series(0.0, index=money_flow.index)
            negative_flow = pd.Series(0.0, index=money_flow.index)
            
            # 计算正/负资金流
            price_diff = typical_price.diff()
            positive_flow[price_diff > 0] = money_flow[price_diff > 0]
            negative_flow[price_diff < 0] = money_flow[price_diff < 0]
            
            # 14日MFI
            mfi_period = 14
            if len(df) > mfi_period:
                positive_sum = positive_flow.rolling(window=mfi_period).sum()
                negative_sum = negative_flow.rolling(window=mfi_period).sum()
                
                # 避免除零
                money_ratio = positive_sum / negative_sum.replace(0, 1e-10)
                df['MFI'] = 100 - (100 / (1 + money_ratio))
            else:
                df['MFI'] = pd.Series(np.nan, index=df.index)
            
            # 计算ATR和波动率
            df['ATR'] = self.calculate_atr(df, self.params['atr_period'])
            df['Volatility'] = df['ATR'] / df['close'] * 100
            
            # 动量指标
            df['ROC'] = df['close'].pct_change(periods=10) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            logger.exception(e)
            raise
            
    def calculate_score(self, df, market_type='A'):
        """计算股票评分 - 适应不同市场特性"""
        try:
            if len(df) < 20:
                logger.warning(f"数据点数({len(df)})不足，无法可靠计算评分")
                return 0
                
            # 根据市场类型选择参数
            self.params = self.market_params.get(market_type, self.market_params['A'])
            
            score = 0
            latest = df.iloc[-1]
            
            # 检查数据是否有足够的历史记录
            if len(df) < max(60, self.params['ma_periods']['long']):
                logger.warning("数据量不足，评分可能不准确")
                
            # 趋势得分 - 不同市场趋势权重调整
            trend_score = 0
            if latest['MA5'] > latest['MA20']:
                trend_score += 10
            if latest['MA20'] > latest['MA60']:
                trend_score += 10
                
            # 短期趋势斜率
            try:
                ma5_slope = (latest['MA5'] - df.iloc[-3]['MA5']) / df.iloc[-3]['MA5']
                # 根据市场类型设置不同阈值
                slope_threshold = self.params['price_change_threshold']
                if ma5_slope > slope_threshold:
                    trend_score += 10
            except:
                pass  # 安全处理
                
            score += trend_score
                
            # 判断市场状态(牛熊市)
            is_bull_market = False
            if len(df) >= 120:  # 至少需要120天数据
                # 简单判断：近120日均线向上为牛市
                ma120 = df['close'].rolling(window=120).mean()
                if ma120.iloc[-1] > ma120.iloc[-20] and df.iloc[-1]['close'] > ma120.iloc[-1]:
                    is_bull_market = True
                
            # RSI得分 - 根据牛熊市调整阈值
            rsi_score = 0
            if pd.notna(latest['RSI']):
                if is_bull_market:
                    # 牛市RSI阈值上移
                    if 50 <= latest['RSI'] <= 60:  # 牛市中轨区间
                        rsi_score = 10
                    elif 40 <= latest['RSI'] < 50 or 60 < latest['RSI'] <= 75:  # 过渡区
                        rsi_score = 15
                    elif 25 <= latest['RSI'] < 40:  # 牛市超卖区间
                        rsi_score = 20
                else:
                    # 熊市RSI阈值下移
                    if 45 <= latest['RSI'] <= 55:  # 熊市中轨区间
                        rsi_score = 10
                    elif 35 <= latest['RSI'] < 45 or 55 < latest['RSI'] <= 65:  # 过渡区
                        rsi_score = 15
                    elif 20 <= latest['RSI'] < 35:  # 熊市超卖区间
                        rsi_score = 20
            score += rsi_score
                
            # MACD得分
            macd_score = 0
            if latest['MACD'] > latest['Signal']:
                macd_score += 10
                
                # 确认是否为最近金叉
                cross_days = 0
                for i in range(2, min(6, len(df))):  # 最多查找过去5天
                    if df.iloc[-i]['MACD'] <= df.iloc[-i]['Signal']:
                        cross_days = i - 1
                        break
                        
                if cross_days > 0:  # 最近n天内金叉
                    macd_score += (5 - cross_days) if cross_days < 5 else 0
                    
            # 柱状图放大
            hist_change = 0
            if len(df) > 2:
                hist_change = (latest['MACD_hist'] - df.iloc[-2]['MACD_hist']) / abs(df.iloc[-2]['MACD_hist'] + 1e-10)
                
            if latest['MACD_hist'] > 0 and hist_change > 0.1:
                macd_score += 5
                
            score += macd_score
                
            # 成交量得分 - 根据市场类型调整阈值
            volume_score = 0
            
            # 根据市场类型设置不同放量标准
            extreme_volume = self.params['volume_ratio_threshold']['extreme']
            normal_volume = self.params['volume_ratio_threshold']['normal']
            
            if latest['Volume_Ratio'] > extreme_volume:
                volume_score += 15
            elif latest['Volume_Ratio'] > normal_volume:
                volume_score += 10
                
            # 量价关系判断
            try:
                price_change = (latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
                
                # 特别处理涨停情况
                is_star_chinext = False
                stock_code = getattr(self, 'current_stock_code', '')
                if stock_code.startswith('688') or stock_code.startswith('300'):
                    is_star_chinext = True
                    
                # 根据不同板块判断涨停情况
                is_limit_up = False
                if is_star_chinext and price_change > 0.195:  # 科创板/创业板20%涨停
                    is_limit_up = True
                elif not is_star_chinext and price_change > 0.095:  # 主板10%涨停
                    is_limit_up = True
                    
                # 判断一字板还是换手涨停
                is_one_way_limit = False
                if is_limit_up and latest['high'] == latest['low']:
                    is_one_way_limit = True
                
                if is_limit_up:
                    # 一字板不看量
                    if is_one_way_limit:
                        volume_score += 10  # 一字涨停加分，但幅度较小
                    # 换手涨停看放量
                    elif latest['Volume_Ratio'] > normal_volume:
                        volume_score += 15  # 换手涨停且放量，显著加分
                # 非涨停情况
                elif price_change > self.params['price_change_threshold'] and latest['Volume_Ratio'] > normal_volume:
                    volume_score += 15  # 量增价涨
                elif price_change < -self.params['price_change_threshold'] and latest['Volume_Ratio'] > normal_volume:
                    volume_score -= 10  # 量增价跌，看空
                    
                # 资金流向(MFI)加分项
                if pd.notna(latest['MFI']):
                    # 正向资金流入
                    if latest['MFI'] > 50 and price_change > 0:
                        volume_score += 5
                    # 见底反转信号
                    elif latest['MFI'] < 20 and latest['RSI'] < 30 and latest['MFI'] > df.iloc[-2]['MFI']:
                        volume_score += 10
            except Exception as e:
                logger.warning(f"量价关系评分计算错误: {str(e)}")
                
            score += volume_score
            
            # 布林带相关得分
            bb_score = 0
            
            # 布林带突破
            if latest['close'] > latest['BB_upper']:
                # 强势突破上轨
                bb_score += 15 if latest['Volume_Ratio'] > normal_volume else 5
            elif latest['close'] < latest['BB_lower']:
                # 弱势跌破下轨
                bb_score -= 10 if latest['Volume_Ratio'] > normal_volume else 5
                
            # 布林带收窄后放大(突破形态)
            if pd.notna(latest['BB_narrowing']) and latest['BB_narrowing'] and latest['Volume_Ratio'] > normal_volume:
                bb_score += 10
                
            score += bb_score
            
            # 限制最终分数范围
            score = max(0, min(score, 100))
                
            return score
                
        except Exception as e:
            logger.error(f"计算评分时出错: {str(e)}")
            logger.exception(e)
            return 0  # 出错时返回0分
            
    def get_ai_analysis(self, df, stock_code, market_type="A", stream=False):
        """使用 OpenAI 进行 AI 分析"""
        try:
            type_name = 'A股' if market_type == 'A' else '港股'
            logger.info(f"开始AI分析中国{type_name}股票 {stock_code}, 流式模式: {stream}")
            recent_data = df.tail(60).to_dict('records')
            
            technical_summary = {
                'trend': 'upward' if df.iloc[-1]['MA5'] > df.iloc[-1]['MA20'] else 'downward',
                'volatility': f"{df.iloc[-1]['Volatility']:.2f}%",
                'volume_trend': 'increasing' if df.iloc[-1]['Volume_Ratio'] > 1 else 'decreasing',
                'rsi_level': df.iloc[-1]['RSI']
            }

            # 生成提示词
            prompt = f"""
            分析中国{type_name}股市场股票 {stock_code}：

            技术指标概要：
            {technical_summary}
            
            近60日交易数据：
            {recent_data}
            
            请提供：
            1. 趋势分析（包含支撑位和压力位）
            2. 成交量分析及其含义
            3. 风险评估（包含波动率分析）
            4. 短期和中期目标价位
            5. 关键技术位分析
            6. 具体交易建议（包含止损位）
            
            请基于技术指标和市场动态进行分析，给出具体数据支持。
            """
            
            logger.debug(f"生成的AI分析提示词: {self._truncate_json_for_logging(prompt, 100)}...")
            
            # 检查API配置
            if not self.API_URL:
                error_msg = "API URL未配置，无法进行AI分析"
                logger.error(error_msg)
                return error_msg if not stream else (yield json.dumps({"error": error_msg}))
                
            if not self.API_KEY:
                error_msg = "API Key未配置，无法进行AI分析"
                logger.error(error_msg)
                return error_msg if not stream else (yield json.dumps({"error": error_msg}))
            
            # 标准化API URL
            api_url = APIUtils.format_api_url(self.API_URL)
            
            logger.debug(f"标准化后的API URL: {api_url}")
            
            # 构建请求头和请求体
            headers = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.API_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # 流式处理设置
            if stream:
                logger.debug(f"配置流式参数，使用API URL: {api_url}")
                payload["stream"] = True  # 明确设置stream参数为True
                
                try:
                    logger.debug(f"发起流式API请求: {api_url}")
                    logger.debug(f"请求载荷: {self._truncate_json_for_logging(payload)}")
                    
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.API_TIMEOUT,  # 增加超时时间
                        stream=True
                    )
                    
                    logger.debug(f"API流式响应状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        logger.info(f"成功获取API流式响应，开始处理")
                        yield from self._process_ai_stream(response, stock_code)
                    else:
                        try:
                            error_response = response.json()
                            error_text = self._truncate_json_for_logging(error_response)
                        except:
                            error_text = response.text[:500] if response.text else "无响应内容"
                            
                        error_msg = f"API请求失败: 状态码 {response.status_code}, 响应: {error_text}"
                        logger.error(error_msg)
                        yield json.dumps({"stock_code": stock_code, "error": error_msg})
                        
                except Exception as e:
                    error_msg = f"流式API请求异常: {str(e)}"
                    logger.error(error_msg)
                    logger.exception(e)
                    yield json.dumps({"stock_code": stock_code, "error": error_msg})
            else:
                # 非流式处理
                logger.debug(f"发起非流式API请求: {api_url}")
                
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.API_TIMEOUT
                    )
                    
                    logger.debug(f"API非流式响应状态码: {response.status_code}")
                    
                    if response.status_code == 200:
                        api_response = response.json()
                        content = api_response['choices'][0]['message']['content']
                        logger.info(f"成功获取AI分析结果，长度: {len(content)}")
                        logger.debug(f"AI分析结果前100字符: {content[:100]}...")
                        return content
                    else:
                        try:
                            error_response = response.json()
                            error_text = self._truncate_json_for_logging(error_response)
                        except:
                            error_text = response.text[:500] if response.text else "无响应内容"
                            
                        error_msg = f"API请求失败: 状态码 {response.status_code}, 响应: {error_text}"
                        logger.error(error_msg)
                        return error_msg
                        
                except Exception as e:
                    error_msg = f"非流式API请求异常: {str(e)}"
                    logger.error(error_msg)
                    logger.exception(e)
                    return error_msg
            
        except Exception as e:
            error_msg = f"AI 分析过程中发生错误: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            
            if stream:
                logger.debug("在流式模式下返回异常信息")
                error_json = json.dumps({"stock_code": stock_code, "error": error_msg})
                logger.info(f"流式异常输出: {error_json}")
                yield error_json
            else:
                return error_msg
    
    def _truncate_json_for_logging(self, json_obj, max_length=500):
        """截断JSON对象用于日志记录，避免日志过大
        
        Args:
            json_obj: 要截断的JSON对象
            max_length: 最大字符长度，默认500
            
        Returns:
            str: 截断后的JSON字符串
        """
        json_str = json.dumps(json_obj, ensure_ascii=False)
        if len(json_str) <= max_length:
            return json_str
        return json_str[:max_length] + f"... [截断，总长度: {len(json_str)}字符]"
    
    def _process_ai_stream(self, response, stock_code) -> Generator[str, None, None]:
        """处理AI流式响应"""
        logger.info(f"开始处理股票 {stock_code} 的AI流式响应")
        buffer = ""
        chunk_count = 0
        
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    
                    # 跳过保持连接的空行
                    if line.strip() == '':
                        logger.debug("跳过空行")
                        continue
                        
                    # 数据行通常以"data: "开头
                    if line.startswith('data: '):
                        data_content = line[6:]  # 移除 "data: " 前缀
                        
                        # 检查是否为流的结束
                        if data_content.strip() == '[DONE]':
                            logger.debug("收到流结束标记 [DONE]")
                            break
                            
                        try:
                            json_data = json.loads(data_content)
                            
                            if 'choices' in json_data:
                                delta = json_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    chunk_count += 1
                                    buffer += content

                                    # 创建包含AI分析片段的JSON
                                    chunk_json = json.dumps({
                                        "stock_code": stock_code,
                                        "ai_analysis_chunk": content
                                    })
                                    yield chunk_json
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误: {str(e)}, 行内容: {data_content}")
                            # 忽略无法解析的JSON
                            pass
                    else:
                        logger.warning(f"收到非'data:'开头的行: {line}")
            
            logger.info(f"AI流式处理完成，共收到 {chunk_count} 个内容片段，总长度: {len(buffer)}")
            
            # 如果buffer不为空，最后一次发送完整内容
            if buffer and not buffer.endswith('\n'):
                logger.debug("发送换行符")
                yield json.dumps({"stock_code": stock_code, "ai_analysis_chunk": "\n"})
        
        except Exception as e:
            error_msg = f"处理AI流式响应时出错: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            yield json.dumps({"stock_code": stock_code, "error": error_msg})

    
    def get_recommendation(self, score):
        """根据得分给出建议"""
        logger.debug(f"根据评分 {score} 生成投资建议")
        if score >= 80:
            return '强烈推荐买入'
        elif score >= 60:
            return '建议买入'
        elif score >= 40:
            return '观望'
        elif score >= 20:
            return '建议卖出'
        else:
            return '强烈建议卖出'
    
    def analyze_stock(self, stock_code, market_type='A', stream=False):
        """分析单个股票"""
        try:
            logger.info(f"开始分析股票: {stock_code}, 市场: {market_type}, 流式模式: {stream}")
            
            # 保存当前股票代码用于指标计算
            self.current_stock_code = stock_code
            
            # 基于市场类型设置参数
            self.params = self.market_params.get(market_type, self.market_params['A'])
            
            # 获取股票数据
            logger.debug(f"获取股票 {stock_code} 数据")
            df = self.get_stock_data(stock_code, market_type)
            
            # 计算技术指标
            logger.debug(f"计算股票 {stock_code} 技术指标")
            df = self.calculate_indicators(df)
            
            # 评分系统
            logger.debug(f"计算股票 {stock_code} 评分")
            score = self.calculate_score(df, market_type)
            logger.info(f"股票 {stock_code} 评分结果: {score}")
            
            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 处理 RSI 的 NaN 值
            rsi_value = latest['RSI']
            if pd.isna(rsi_value):
                rsi_value = None  
            
            # 生成报告（保持原有格式）
            report = {
                'stock_code': stock_code,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'score': score,
                'price': latest['close'],
                'price_change': (latest['close'] - prev['close']) / prev['close'] * 100,
                'ma_trend': 'UP' if latest['MA5'] > latest['MA20'] else 'DOWN',
                'rsi': rsi_value,  # 使用处理后的 RSI 值
                'macd_signal': 'BUY' if latest['MACD'] > latest['Signal'] else 'SELL',
                'volume_status': 'HIGH' if latest['Volume_Ratio'] > 1.5 else 'NORMAL',
                'recommendation': self.get_recommendation(score)
            }
            logger.debug(f"生成股票 {stock_code} 基础报告: {self._truncate_json_for_logging(report, 100)}...")
            
            if stream:
                logger.info(f"以流式模式返回股票 {stock_code} 分析结果")
                # 先返回基本报告结构
                base_report = dict(report)
                base_report['ai_analysis'] = ''
                base_report_json = json.dumps(base_report)
                logger.debug(f"基础报告JSON: {self._truncate_json_for_logging(base_report_json, 100)}...")
                logger.info(f"发送基础报告: {base_report_json}")
                yield base_report_json
                
                # 然后流式返回AI分析部分
                logger.debug(f"开始获取股票 {stock_code} 的流式AI分析")
                ai_chunks_count = 0
                for ai_chunk in self.get_ai_analysis(df, stock_code, stream=True):
                    ai_chunks_count += 1
                    yield ai_chunk
                logger.info(f"股票 {stock_code} 流式AI分析完成，共发送 {ai_chunks_count} 个块")
            else:
                logger.info(f"以非流式模式返回股票 {stock_code} 分析结果")
                logger.debug(f"开始获取股票 {stock_code} 的AI分析")
                report['ai_analysis'] = self.get_ai_analysis(df, stock_code)
                logger.debug(f"AI分析结果长度: {len(report['ai_analysis'])}")
                return report
            
        except Exception as e:
            error_msg = f"分析股票 {stock_code} 时出错: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            
            if stream:
                error_json = json.dumps({'stock_code': stock_code, 'error': error_msg})
                logger.info(f"流式错误输出: {error_json}")
                yield error_json
            else:
                raise
            
    def scan_market(self, stock_list, min_score=60, market_type='A', stream=False):
        """扫描市场，寻找符合条件的股票"""
        logger.info(f"开始扫描市场，股票数量: {len(stock_list)}, 最低分数: {min_score}, 市场: {market_type}, 流式模式: {stream}")
        
        if not stream:
            recommendations = []
            
            for stock_code in stock_list:
                try:
                    logger.debug(f"分析股票: {stock_code}")
                    report = self.analyze_stock(stock_code, market_type)
                    if report['score'] >= min_score:
                        logger.info(f"股票 {stock_code} 评分 {report['score']} >= {min_score}，添加到推荐列表")
                        recommendations.append(report)
                    else:
                        logger.debug(f"股票 {stock_code} 评分 {report['score']} < {min_score}，不添加到推荐列表")
                except Exception as e:
                    logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
                    logger.exception(e)
                    continue
                    
            # 按得分排序
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"扫描完成，找到 {len(recommendations)} 个推荐股票")
            return recommendations
        else:
            # 流式处理每个股票
            logger.info(f"开始流式扫描 {len(stock_list)} 只股票")
            stock_count = 0
            for stock_code in stock_list:
                stock_count += 1
                logger.debug(f"流式分析股票 {stock_code} ({stock_count}/{len(stock_list)})")
                try:
                    # 分析单只股票并获取流式结果
                    chunk_count = 0
                    for chunk in self.analyze_stock(stock_code, market_type, stream=True):
                        chunk_count += 1
                        yield chunk
                    logger.debug(f"股票 {stock_code} 流式分析完成，共 {chunk_count} 个块")
                except Exception as e:
                    error_msg = f"分析股票 {stock_code} 时出错: {str(e)}"
                    logger.error(error_msg)
                    logger.exception(e)
                    error_json = json.dumps({'stock_code': stock_code, 'error': error_msg})
                    logger.info(f"流式错误输出: {error_json}")
                    yield error_json
            logger.info(f"流式扫描完成，处理了 {stock_count} 只股票")

