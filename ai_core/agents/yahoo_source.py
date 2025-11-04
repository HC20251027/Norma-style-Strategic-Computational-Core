"""
Yahoo Finance data source implementation
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("yahoo_finance_source")


class YahooFinanceSource(BaseAPI):
    """Yahoo Finance API data source implementation"""

    def __init__(self, config: Dict[str, Any], proxy_url: Optional[str] = None):
        """Initialize Yahoo Finance data source

        Args:
            config: Configuration dictionary containing API settings
        """
        self._timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        if proxy_url:
            self.proxy_url = proxy_url
        self.headers = {
            "X-Original-Host": config["yahoo_base_url"],
            "X-Biz-Id": "matrix-agent",
            "X-Request-Timeout": str(config["timeout"] - 5),
        }

    @property
    def source_name(self) -> str:
        """Get the data source name

        Returns:
            str: Name of the data source
        """
        return "yahoo_finance"

    def get_api_info(self) -> Dict[str, Any]:
        """Get basic information about the data source

        Returns:
            Dict[str, Any]: Dictionary containing basic information like name and version
        """
        return {
            "name": self.source_name,
            "description": "Yahoo Finance data source, providing stock price and company information query and stock related news query",
        }

    async def get_stock_price(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        events: str = "",
    ) -> Dict[str, Any]:
        """Get stock price data. Please set start_date, end_date, interval reasonably to avoid getting too much data,
        which could cause request timeout or performance issues.

        Args:
            symbol: Stock code
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Time interval, options: 1m|2m|5m|15m|30m|60m|1d|1wk|1mo, default: 1d
            events: Event type, options: capitalGain|div|split|earn|history, default: empty

        Returns:
            Dict[str, Any]: Dictionary containing stock price data, e.g.
            {
                "success": True,                   # Whether successful
                "data": {                          # If successful, contains following fields
                    "symbol": "AAPL",              # Stock code
                    "prices": [                     # Price list, chronological order
                        {
                            "date": "2024-01-01",  # Date
                            "open": 182.15,        # Opening price
                            "high": 185.10,        # Highest price
                            "low": 181.80,         # Lowest price
                            "close": 184.25,       # Closing price
                            "volume": 32456789     # Trading volume
                        },
                        {
                            "date": "2024-01-02",
                            "open": 184.30,
                            "high": 186.20,
                            "low": 183.95,
                            "close": 185.75,
                            "volume": 28975632
                        }
                    ]
                }
            }
        """
        try:
            # Convert date string to timestamp
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

            if start_timestamp > end_timestamp:
                raise ValueError("start_date cannot be greater than end_date")

            # Build request parameters
            params = {
                "symbol": symbol,
                "period1": start_timestamp,
                "period2": end_timestamp,
                "interval": interval,
                "region": "US",  # Default use US area
                "includePrePost": "false",
                "useYfid": "true",
                "includeAdjustedClose": "true",
            }

            # If events parameter is provided, add to request
            if events:
                params["events"] = events

            request_url = f"{self.proxy_url}/stock/v3/get-chart"

            # Send request using aiohttp
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                    response.raise_for_status()
                    # Parse the response
                    data = await response.json()

            # Check if there is an error in API response
            if data.get("chart", {}).get("error"):
                return {"success": False, "error": str(data["chart"]["error"])}

            # Parse response data
            chart_data = data["chart"]["result"][0]
            timestamps = chart_data["timestamp"]
            quote = chart_data["indicators"]["quote"][0]

            # Build price data list
            prices = []
            for i, timestamp in enumerate(timestamps):
                price_data = {
                    "date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d"),
                    "open": quote["open"][i],
                    "high": quote["high"][i],
                    "low": quote["low"][i],
                    "close": quote["close"][i],
                    "volume": int(quote["volume"][i]),
                }
                prices.append(price_data)

            return {"success": True, "data": {"symbol": symbol, "prices": prices}}

        except asyncio.TimeoutError:
            error_msg = f"Request timeout (timeout={self._timeout}s)"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except aiohttp.ClientError as e:
            error_msg = f"HTTP request error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            logger.error(f"Error occurred while getting stock price data: {str(e)}")
            logger.exception(e)
            return {"success": False, "error": f"Unknown error: {str(e)}"}

    async def get_stock_news(self, symbol: str, region: str = "US", snippet_count: int = 10) -> Dict[str, Any]:
        """获取股票相关的新闻数据
        Args:
            symbol(str): Stock code
            region(str): Region code, defaults to US
            snippet_count(int): Number of news items to return, defaults to 10
        Returns:
            Dict[str, Any]: Dictionary containing stock news data, e.g.
            {
                "success": True,
                "data": {
                    "symbol": "AAPL",
                    "simple_news": [
                        {
                            "title": "标题",
                            "publisher": "发布者",
                            "publish_date": "发布时间",
                            "link": "链接",
                            "uuid": "UUID",
                            "content_type": "类型",
                            "thumbnail": "缩略图URL",
                            "tickers": ["AAPL", "MSFT"]
                        }
                    ]
                }
            }
        """
        try:
            # 构建请求URL - 使用正确的新闻API端点
            request_url = f"{self.proxy_url}/news/v2/list"

            # 构建请求参数
            params = {"region": region, "snippetCount": str(snippet_count), "s": symbol}

            # 发送POST请求
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    # 使用POST请求，并设置空数据体
                    async with session.post(
                        request_url,
                        headers=self.headers,
                        params=params,
                        data="",  # load_more 逻辑，先不适配
                        timeout=self._timeout,
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

                        # 提取并处理新闻数据 - 根据实际响应格式调整
                        stream_items = []
                        # 检查响应结构中的main.stream路径
                        if data.get("data") and data["data"].get("main") and data["data"]["main"].get("stream"):
                            stream_items = data["data"]["main"]["stream"]

                        # 转换为简化的新闻对象列表
                        simple_news = []
                        for stream_item in stream_items:
                            content = stream_item.get("content", {})
                            if not content:
                                continue

                            # 获取链接
                            link = ""
                            click_through_url = content.get("clickThroughUrl", {})
                            if click_through_url and click_through_url.get("url"):
                                link = click_through_url["url"]

                            # 获取发布者
                            publisher = ""
                            if content.get("provider") and content["provider"].get("displayName"):
                                publisher = content["provider"]["displayName"]

                            # 创建简化的新闻项
                            news_item = {
                                "title": content.get("title", ""),
                                "publisher": publisher,
                                "publish_date": content.get("pubDate", ""),
                                "link": link,
                                "uuid": content.get("id", ""),
                                "content_type": content.get("contentType", ""),
                                "thumbnail": self._extract_thumbnail(content.get("thumbnail", {})),
                                "tickers": self._extract_tickers(content.get("finance", {})),
                            }
                            simple_news.append(news_item)

                        # 返回结构化的新闻列表
                        return {"success": True, "data": {"symbol": symbol, "simple_news": simple_news}}

            except asyncio.TimeoutError:
                error_msg = f"请求超时 (timeout={self._timeout}秒)"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            except aiohttp.ClientError as e:
                error_msg = f"HTTP请求错误: {str(e)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            except Exception as e:
                error_msg = f"获取股票新闻信息时发生错误: {str(e)}"
                logger.error(error_msg)
                logger.exception(e)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"获取股票新闻信息时发生错误: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    def _extract_thumbnail(self, thumbnail_data: Dict[str, Any]) -> str:
        """从缩略图数据中提取第一个可用的URL

        Args:
            thumbnail_data: 包含缩略图信息的字典

        Returns:
            str: 缩略图URL，如果没有则返回空字符串
        """
        if not thumbnail_data or not thumbnail_data.get("resolutions"):
            return ""

        # 尝试获取原始尺寸缩略图
        for resolution in thumbnail_data["resolutions"]:
            if resolution.get("tag") == "original" and resolution.get("url"):
                return resolution["url"]

        # 如果没有原始尺寸，则获取第一个可用的URL
        if thumbnail_data["resolutions"] and thumbnail_data["resolutions"][0].get("url"):
            return thumbnail_data["resolutions"][0]["url"]

        return ""

    def _extract_tickers(self, finance_data: Dict[str, Any]) -> List[str]:
        """从金融数据中提取股票代码列表

        Args:
            finance_data: 包含金融信息的字典

        Returns:
            List[str]: 股票代码列表
        """
        tickers = []
        if finance_data and finance_data.get("stockTickers"):
            for ticker_data in finance_data["stockTickers"]:
                if ticker_data.get("symbol"):
                    tickers.append(ticker_data["symbol"])
        return tickers

    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic stock information

        Args:
            symbol(str): Stock code. For Hong Kong stocks, use 4-digit format like 1211.HK (not 01211.HK). For Chinese stocks, use 6-digit format with .SS suffix for Shanghai stocks (e.g. 600009.SS) and .SZ suffix for Shenzhen stocks (e.g. 000002.SZ).

        Returns:
            Dict[str, Any]: Dictionary containing basic stock information, e.g.
            {
                "success": True,                  # Whether successful
                "data": {                         # If successful, contains following fields
                    "symbol": "AAPL",             # Stock code
                    "market_cap": 2850000000000,  # Market capitalization
                    "pe_ratio": 31.25,            # Trailing P/E ratio
                    "forward_pe": 28.4,           # Forward P/E ratio
                    "dividend_yield": 0.0052,     # Dividend yield
                    "beta": 1.28,                 # Beta coefficient
                    "fifty_two_week": {           # 52-week data
                        "low": 148.5,             # 52-week lowest price
                        "high": 199.62            # 52-week highest price
                    },
                    "moving_averages": {          # Moving averages
                        "fifty_day": 182.45,      # 50-day average price
                        "two_hundred_day": 178.30 # 200-day average price
                    },
                    "volume": {                   # Trading volume data
                        "current": 45678912,      # Current trading volume
                        "average": 52456789       # Average trading volume
                    }
                }
            }
        """

        try:
            # Build request parameters
            params = {"symbol": symbol, "modules": "summaryDetail"}

            request_url = f"{self.proxy_url}/stock/get-fundamentals"

            # Send request
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        response.raise_for_status()
                        data = await response.json()

            except asyncio.TimeoutError:
                error_msg = f"Request timeout (timeout={self._timeout}s)"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            except aiohttp.ClientError as e:
                error_msg = f"HTTP request error: {str(e)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            # Check if there is an error in API response
            if data.get("quoteSummary", {}).get("error"):
                error_msg = str(data["quoteSummary"]["error"])
                logger.error(f"API returned error: {error_msg}")
                return {"success": False, "error": error_msg}

            # Parse response data
            summary_detail = data["quoteSummary"]["result"][0]["summaryDetail"]

            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "market_cap": float(summary_detail.get("marketCap", {}).get("raw", 0)),
                    "pe_ratio": float(summary_detail.get("trailingPE", {}).get("raw", 0)),
                    "forward_pe": float(summary_detail.get("forwardPE", {}).get("raw", 0)),
                    "dividend_yield": float(summary_detail.get("dividendYield", {}).get("raw", 0)),
                    "beta": float(summary_detail.get("beta", {}).get("raw", 0)),
                    "fifty_two_week": {
                        "low": float(summary_detail.get("fiftyTwoWeekLow", {}).get("raw", 0)),
                        "high": float(summary_detail.get("fiftyTwoWeekHigh", {}).get("raw", 0)),
                    },
                    "moving_averages": {
                        "fifty_day": float(summary_detail.get("fiftyDayAverage", {}).get("raw", 0)),
                        "two_hundred_day": float(summary_detail.get("twoHundredDayAverage", {}).get("raw", 0)),
                    },
                    "volume": {
                        "current": int(summary_detail.get("volume", {}).get("raw", 0)),
                        "average": int(summary_detail.get("averageVolume", {}).get("raw", 0)),
                    },
                },
            }

        except Exception as e:
            error_msg = f"Error occurred while getting stock financial data: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def get_multiple_stocks_price(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        events: str = "",
    ) -> Dict[str, Any]:
        """Get price data for multiple stocks

        Args:
            symbols(List[str]): Stock code list
            start_date(str): Start date in YYYY-MM-DD format
            end_date(str): End date in YYYY-MM-DD format
            interval(str): Time interval, options: 1m|2m|5m|15m|30m|60m|1d|1wk|1mo, default: 1d
            events(str): Event type, options: capitalGain|div|split|earn|history, default: empty

        Returns:
            Dict[str, Any]: Dictionary containing stock price data, e.g.
            {
                "success": true,               # Whether successful
                "data": {                      # If successful, contains following fields
                    "count": 2,                # Number of stocks
                    "stocks": [                # Stock data list
                        {
                            "symbol": "AAPL",  # Stock code
                            "prices": [        # Price list
                                {
                                    "date": "2024-01-01",  # Date
                                    "open": 182.15,        # Opening price
                                    "high": 185.10,        # Highest price
                                    "low": 181.80,         # Lowest price
                                    "close": 184.25,       # Closing price
                                    "volume": 32456789     # Trading volume
                                }
                            ]
                        },
                        {
                            "symbol": "GOOGL",
                            "prices": [
                                {
                                    "date": "2024-01-01",
                                    "open": 138.56,
                                    "high": 139.20,
                                    "low": 137.95,
                                    "close": 138.85,
                                    "volume": 18654123
                                }
                            ]
                        }
                    ],
                    "failed_symbols": []       # Failed stock information
                }
            }
        """

        try:
            stocks_data = []
            failed_symbols = []

            # Iterate over stock code list to get data for each stock
            for symbol in symbols:
                try:
                    result = await self.get_stock_price(
                        symbol=symbol, start_date=start_date, end_date=end_date, interval=interval, events=events
                    )
                    if result["success"]:
                        stocks_data.append(result["data"])
                    else:
                        failed_symbols.append((symbol, result["error"]))
                        logger.warning(f"Failed to get data for stock {symbol}: {result['error']}")
                except Exception as e:
                    failed_symbols.append((symbol, str(e)))
                    logger.error(f"Error occurred while getting data for stock {symbol}: {str(e)}")
                    logger.exception(e)

            # If all stocks fail to get data
            if len(failed_symbols) == len(symbols):
                error_msg = "All stock data retrieval failed:\n" + "\n".join([f"{symbol}: {error}" for symbol, error in failed_symbols])
                return {"success": False, "error": error_msg}

            # Return successfully obtained data, including failed information
            return {
                "success": True,
                "data": {
                    "count": len(stocks_data),
                    "stocks": stocks_data,
                    "failed_symbols": [{"symbol": symbol, "error": error} for symbol, error in failed_symbols] if failed_symbols else [],
                },
            }

        except Exception as e:
            logger.error(f"Error occurred while batch getting stock data: {str(e)}")
            logger.exception(e)
            return {"success": False, "error": str(e)}

    async def get_stock_insights(self, symbol: str) -> Dict[str, Any]:
        """Get stock insight data, including technical analysis, valuation, and company snapshot

        Args:
            symbol(str): Stock code

        Returns:
            Dict[str, Any]: Dictionary containing stock insight data, e.g.
            {
                "success": true,                # Whether successful
                "data": {                       # If successful, contains following fields
                    "symbol": "AAPL",           # Stock code
                    "technical_analysis": {     # Technical analysis data
                        "short_term": {         # Short-term outlook
                            "direction": "Bullish",  # Direction (Bullish/Bearish)
                            "score": 4,              # Score, 1-5
                            "description": "Strong upward momentum"  # Description
                        },
                        "support": 175.80,      # Support level
                        "resistance": 198.50,   # Resistance level
                        "stop_loss": 172.40,    # Stop loss level
                        "provider": "Trading Central"  # Technical analysis provider
                    },
                    "valuation": {              # Valuation data
                        "description": "Fairly valued with moderately positive outlook",  # Valuation description
                        "discount": "2.5%",     # Target valuation discount
                        "relative_value": "Premium to sector",  # Relative value
                        "provider": "Morningstar"  # Valuation provider
                    },
                    "company_snapshot": {       # Company snapshot
                        "innovativeness": 0.85,  # Innovativeness score, 0-1
                        "sustainability": 0.72,  # Sustainability score, 0-1
                        "insider_sentiments": 0.65,  # Insider sentiment score, 0-1
                        "earningsReports": 0.90,  # Earnings report score, 0-1
                        "dividends": 0.68        # Dividend score, 0-1
                    },
                    "recommendation": {         # Analyst recommendation
                        "target_price": 205.75,  # Target price
                        "rating": "buy",         # Rating: 'buy' | 'sell' | 'hold'
                        "provider": "Zacks"      # Recommendation provider
                    }
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Start to get_stock_insights of AAPL")
        #     >>> result = client.yahoo_finance.get_stock_insights("AAPL")
        #     >>> if not result["success"]:
        #     ...     print(f"Failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Success")
        # """
        try:
            # Build request URL
            request_url = f"{self.proxy_url}/stock/v3/get-insights"

            # Build request parameters
            params = {"symbol": symbol}

            # Send request
            async with aiohttp.ClientSession(trust_env=True) as session:
                try:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        # Check response status
                        response.raise_for_status()
                        data = await response.json()
                except asyncio.TimeoutError:
                    return {"success": False, "error": f"Request timeout (timeout={self._timeout}s)"}
                except aiohttp.ClientError as e:
                    return {"success": False, "error": f"HTTP request error: {str(e)}"}

            # Check if there is an error in API response
            if data.get("finance", {}).get("error"):
                return {"success": False, "error": str(data["finance"]["error"])}

            # Parse response data
            result = data["finance"]["result"]
            instrument_info = result.get("instrumentInfo", {})
            technical_events = instrument_info.get("technicalEvents", {})
            key_technicals = instrument_info.get("keyTechnicals", {})
            valuation = instrument_info.get("valuation", {})
            company_snapshot = result.get("companySnapshot", {}).get("company", {})
            recommendation = result.get("recommendation", {})

            # Build return data
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "technical_analysis": {
                        "short_term": {
                            "direction": technical_events.get("shortTermOutlook", {}).get("direction", ""),
                            "score": technical_events.get("shortTermOutlook", {}).get("score", 0),
                            "description": technical_events.get("shortTermOutlook", {}).get("stateDescription", ""),
                        },
                        "support": float(key_technicals.get("support", 0)),
                        "resistance": float(key_technicals.get("resistance", 0)),
                        "stop_loss": float(key_technicals.get("stopLoss", 0)),
                        "provider": technical_events.get("provider", ""),
                    },
                    "valuation": {
                        "description": valuation.get("description", ""),
                        "discount": valuation.get("discount", ""),
                        "relative_value": valuation.get("relativeValue", ""),
                        "provider": valuation.get("provider", ""),
                    },
                    "company_snapshot": {
                        "innovativeness": float(company_snapshot.get("innovativeness", 0)),
                        "sustainability": float(company_snapshot.get("sustainability", 0)),
                        "insider_sentiments": float(company_snapshot.get("insiderSentiments", 0)),
                        "earningsReports": float(company_snapshot.get("earningsReports", 0)),
                        "dividends": float(company_snapshot.get("dividends", 0)),
                    },
                    "recommendation": {
                        "target_price": float(recommendation.get("targetPrice", 0)),
                        "rating": recommendation.get("rating", ""),
                        "provider": recommendation.get("provider", ""),
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error occurred while getting stock insight data: {str(e)}")
            logger.exception(e)
            return {"success": False, "error": str(e)}

    async def get_stock_statistics(self, symbol: str, region: Optional[str] = None, lang: Optional[str] = None) -> Dict[str, Any]:
        """Get stock statistics data, including valuation metrics, financial ratios, and shareholder information

        Args:
            symbol(str): Stock code
            region(str): Region code, options: US, HK, CN, etc.
            lang(str): Language code, options: en-US, zh-CN, etc.

        Returns:
            Dict[str, Any]: Dictionary containing stock statistics data, e.g.
            {
                "success": true,                  # Whether successful
                "data": {                         # If successful, contains following fields
                    "symbol": "AAPL",             # Stock code
                    "valuation_metrics": {        # Valuation metrics
                        "enterprise_value": 2728000000000,  # Enterprise value
                        "forward_pe": 28.4,       # Forward P/E ratio
                        "forward_eps": 6.58,      # Forward EPS
                        "price_to_book": 46.2,    # Price-to-book ratio
                        "enterprise_to_revenue": 7.5,  # Enterprise value-to-revenue ratio
                        "enterprise_to_ebitda": 20.8   # Enterprise value/EBITDA
                    },
                    "profitability": {            # Profitability metrics
                        "most_recent_quarter": "2023-12-31",  # Most recent quarter
                        "net_income": 33915000000,  # Net income
                        "profit_margins": 0.253,  # Profit margin
                        "earnings_growth": 0.125,  # Quarterly earnings growth
                        "revenue_growth": 0.072   # Quarterly revenue growth
                    },
                    "stock_metrics": {            # Stock metrics
                        "beta": 1.28,             # Beta coefficient
                        "year_change": 0.325,     # 52-week change
                        "sp500_year_change": 0.235  # S&P 500 52-week change
                    },
                    "share_statistics": {         # Share statistics
                        "shares_outstanding": 15634100000,  # Total shares outstanding
                        "float_shares": 15627500000,  # Float shares
                        "held_percent_insiders": 0.0059,  # Insider holding percentage
                        "held_percent_institutions": 0.5924,  # Institution holding percentage
                        "short_ratio": 1.85,      # Short ratio
                        "short_percent_of_float": 0.0068  # Short percentage of float
                    },
                    "dividends": {                # Dividend information
                        "last_dividend_value": 0.24,  # Last dividend amount
                        "last_dividend_date": "2024-02-09"  # Last dividend date
                    }
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Start to get_stock_statistics of AAPL")
        #     >>> result = client.yahoo_finance.get_stock_statistics("AAPL")
        #     >>> if not result["success"]:
        #     ...     print(f"Failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Success")
        # """
        try:
            # Build request URL
            request_url = f"{self.proxy_url}/stock/v4/get-statistics"

            # Build request parameters
            params = {
                "symbol": symbol,
            }

            if region:
                params["region"] = region
            if lang:
                params["lang"] = lang

            # Send request
            async with aiohttp.ClientSession(trust_env=True) as session:
                try:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        # Check response status
                        response.raise_for_status()
                        data = await response.json()
                except asyncio.TimeoutError:
                    return {"success": False, "error": f"Request timeout (timeout={self._timeout}s)"}
                except aiohttp.ClientError as e:
                    return {"success": False, "error": f"HTTP request error: {str(e)}"}

            # Check if there is an error in API response
            if data.get("quoteSummary", {}).get("error"):
                return {"success": False, "error": str(data["quoteSummary"]["error"])}

            # Parse response data
            stats = data["quoteSummary"]["result"][0]["defaultKeyStatistics"]

            # Build return data
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "valuation_metrics": {
                        "enterprise_value": float(stats.get("enterpriseValue", {}).get("raw", 0)),
                        "forward_pe": float(stats.get("forwardPE", {}).get("raw", 0)),
                        "forward_eps": float(stats.get("forwardEps", {}).get("raw", 0)),
                        "price_to_book": float(stats.get("priceToBook", {}).get("raw", 0)),
                        "enterprise_to_revenue": float(stats.get("enterpriseToRevenue", {}).get("raw", 0)),
                        "enterprise_to_ebitda": float(stats.get("enterpriseToEbitda", {}).get("raw", 0)),
                    },
                    "profitability": {
                        "most_recent_quarter": stats.get("mostRecentQuarter", {}).get("fmt", ""),
                        "net_income": float(stats.get("netIncomeToCommon", {}).get("raw", 0)),
                        "profit_margins": float(stats.get("profitMargins", {}).get("raw", 0)),
                        "earnings_growth": float(stats.get("earningsQuarterlyGrowth", {}).get("raw", 0)),
                        "revenue_growth": float(stats.get("revenueQuarterlyGrowth", {}).get("raw", 0)),
                    },
                    "stock_metrics": {
                        "beta": float(stats.get("beta", {}).get("raw", 0)),
                        "year_change": float(stats.get("52WeekChange", {}).get("raw", 0)),
                        "sp500_year_change": float(stats.get("SandP52WeekChange", {}).get("raw", 0)),
                    },
                    "share_statistics": {
                        "shares_outstanding": float(stats.get("sharesOutstanding", {}).get("raw", 0)),
                        "float_shares": float(stats.get("floatShares", {}).get("raw", 0)),
                        "held_percent_insiders": float(stats.get("heldPercentInsiders", {}).get("raw", 0)),
                        "held_percent_institutions": float(stats.get("heldPercentInstitutions", {}).get("raw", 0)),
                        "short_ratio": float(stats.get("shortRatio", {}).get("raw", 0)),
                        "short_percent_of_float": float(stats.get("shortPercentOfFloat", {}).get("raw", 0)),
                    },
                    "dividends": {
                        "last_dividend_value": float(stats.get("lastDividendValue", {}).get("raw", 0)),
                        "last_dividend_date": stats.get("lastDividendDate", {}).get("fmt", ""),
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error occurred while getting stock statistics data: {str(e)}")
            logger.exception(e)
            return {"success": False, "error": str(e)}

    async def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """Get stock financial data

        Args:
            symbol(str): Stock code

        Returns:
            Dict[str, Any]: Dictionary containing stock financial data, e.g.
            {
                "success": true,                  # Whether successful
                "data": {                         # If successful, contains following fields
                    "symbol": "AAPL",             # Stock code
                    "price": {                    # Price information
                        "current": 187.45,        # Current price
                        "target": {               # Target price
                            "low": 160.00,        # Lowest target price
                            "high": 240.00,       # Highest target price
                            "mean": 205.75,       # Mean target price
                            "median": 198.50      # Median target price
                        }
                    },
                    "recommendation": {           # Analyst recommendation
                        "mean": 1.8,              # Average recommendation rating (1-5)
                        "key": "buy",             # Recommendation keyword
                        "analysts_count": 35      # Number of analysts
                    },
                    "financial_metrics": {        # Financial metrics
                        "total_cash": 67230000000,  # Total cash
                        "cash_per_share": 4.30,   # Cash per share
                        "total_debt": 111060000000,  # Total debt
                        "debt_to_equity": 175.8,  # Debt-to-equity ratio
                        "current_ratio": 1.02,    # Current ratio
                        "quick_ratio": 0.96       # Quick ratio
                    },
                    "profitability": {            # Profitability metrics
                        "gross_margin": 0.4452,   # Gross margin
                        "operating_margin": 0.3136,  # Operating margin
                        "profit_margin": 0.2530,  # Profit margin
                        "ebitda_margin": 0.3345   # EBITDA margin
                    },
                    "growth": {                   # Growth metrics
                        "revenue_growth": 0.0720,  # Revenue growth
                        "earnings_growth": 0.1250  # Earnings growth
                    },
                    "returns": {                  # Return metrics
                        "return_on_assets": 0.2156,  # Return on assets
                        "return_on_equity": 0.4725   # Return on equity
                    },
                    "cash_flow": {                # Cash flow metrics
                        "operating": 127945000000,  # Operating cash flow
                        "free": 99578000000       # Free cash flow
                    },
                    "currency": "USD"             # Currency of financial data
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Start to get_financial_data of AAPL")
        #     >>> result = client.yahoo_finance.get_financial_data("AAPL")
        #     >>> if not result["success"]:
        #     ...     print(f"Failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Success")
        # """
        try:
            # Build request parameters
            params = {
                "symbol": symbol,
                "region": "US",  # Default use US area
                "lang": "en-US",
                "modules": "financialData",
            }

            request_url = f"{self.proxy_url}/stock/get-fundamentals"

            # Send request
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        response.raise_for_status()
                        data = await response.json()

            except asyncio.TimeoutError:
                error_msg = f"Request timeout (timeout={self._timeout}s)"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            except aiohttp.ClientError as e:
                error_msg = f"Request failed: {str(e)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            # Check if there is an error in API response
            if data.get("quoteSummary", {}).get("error"):
                error_msg = str(data["quoteSummary"]["error"])
                logger.error(f"API returned error: {error_msg}")
                return {"success": False, "error": error_msg}

            # Parse response data
            financial_data = data["quoteSummary"]["result"][0]["financialData"]

            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "price": {
                        "current": float(financial_data.get("currentPrice", {}).get("raw", 0)),
                        "target": {
                            "low": float(financial_data.get("targetLowPrice", {}).get("raw", 0)),
                            "high": float(financial_data.get("targetHighPrice", {}).get("raw", 0)),
                            "mean": float(financial_data.get("targetMeanPrice", {}).get("raw", 0)),
                            "median": float(financial_data.get("targetMedianPrice", {}).get("raw", 0)),
                        },
                    },
                    "recommendation": {
                        "mean": float(financial_data.get("recommendationMean", {}).get("raw", 0)),
                        "key": financial_data.get("recommendationKey", ""),
                        "analysts_count": int(financial_data.get("numberOfAnalystOpinions", {}).get("raw", 0)),
                    },
                    "financial_metrics": {
                        "total_cash": float(financial_data.get("totalCash", {}).get("raw", 0)),
                        "cash_per_share": float(financial_data.get("totalCashPerShare", {}).get("raw", 0)),
                        "total_debt": float(financial_data.get("totalDebt", {}).get("raw", 0)),
                        "debt_to_equity": float(financial_data.get("debtToEquity", {}).get("raw", 0)),
                        "current_ratio": float(financial_data.get("currentRatio", {}).get("raw", 0)),
                        "quick_ratio": float(financial_data.get("quickRatio", {}).get("raw", 0)),
                    },
                    "profitability": {
                        "gross_margin": float(financial_data.get("grossMargins", {}).get("raw", 0)),
                        "operating_margin": float(financial_data.get("operatingMargins", {}).get("raw", 0)),
                        "profit_margin": float(financial_data.get("profitMargins", {}).get("raw", 0)),
                        "ebitda_margin": float(financial_data.get("ebitdaMargins", {}).get("raw", 0)),
                    },
                    "growth": {
                        "revenue_growth": float(financial_data.get("revenueGrowth", {}).get("raw", 0)),
                        "earnings_growth": float(financial_data.get("earningsGrowth", {}).get("raw", 0)),
                    },
                    "returns": {
                        "return_on_assets": float(financial_data.get("returnOnAssets", {}).get("raw", 0)),
                        "return_on_equity": float(financial_data.get("returnOnEquity", {}).get("raw", 0)),
                    },
                    "cash_flow": {
                        "operating": float(financial_data.get("operatingCashflow", {}).get("raw", 0)),
                        "free": float(financial_data.get("freeCashflow", {}).get("raw", 0)),
                    },
                    "currency": financial_data.get("financialCurrency", "USD"),
                },
            }

        except Exception as e:
            error_msg = f"Error occurred while getting stock financial data: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}
