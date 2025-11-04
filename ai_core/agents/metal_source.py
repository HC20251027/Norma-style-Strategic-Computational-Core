"""
Metal data source implementation
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("metal_source")


class MetalSource(BaseAPI):
    """Metal price data source based on Metal API"""

    def __init__(self, config: Dict[str, Any], proxy_url: Optional[str] = None):
        """Initialize Metal price data source

        Args:
            config: Configuration dictionary containing API settings
            proxy_url: Optional proxy URL to override the default from config
        """
        self._timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        if proxy_url:
            self.proxy_url = proxy_url
        self._headers = {
            "X-Original-Host": config["metal_base_url"],
            "X-Biz-Id": "matrix-agent",
            "X-Request-Timeout": str(config["timeout"] - 5),
        }

    @property
    def source_name(self) -> str:
        return "metal"

    def get_api_info(self) -> Dict[str, Any]:
        """Get data source information"""
        return {
            "name": self.source_name,
            "description": "Metal price data source, provides price information for metals such as Gold, Silver, Platinum, Palladium, Rhodium.",
        }

    async def get_metal_price(
        self,
        currency_code: str,
    ) -> Dict[str, Any]:
        """
        Get metal price.

        This method uses the Metal API to get metal prices.

        Args:
            currency_code(str): Currency code, e.g. "USD"

        Returns:
            Dict[str, Any]: Dictionary containing the search results, e.g.
            {
                "success": True,               # Whether successful
                "data": {                      # If successful, contains the following fields
                    "gold": { # Metal type
                        "currency": "USD", # Currency
                        "name": "Gold", # Name
                        "bid": 3318.2999999999997, # Bid price
                        "mid": 3319.2999999999997, # Ask price
                        "high": 3373.6, # Highest price
                        "low": 3264.2, # Lowest price
                        "originalTime": "2025-04-25 17:00:00", # Time
                        "unit": "OUNCE" # Unit
                    }
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> result = await client.metal.get_metal_price(
        #     ...     currency_code="USD"
        #     ... )
        #     >>> if result["success"]:
        #     ...     print(f"Successfully got metal price")
        #     ... else:
        #     ...     print(f"Failed to get metal price: {result['error']}")
        try:
            # Build query parameters
            params = {"currency": currency_code}

            payload = {}

            request_url = f"{self.proxy_url}/web-crawling/api/gold-index"

            # Send request using aiohttp
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(request_url, headers=self._headers, params=params, json=payload, timeout=self._timeout) as response:
                    response.raise_for_status()
                    # Parse the response
                    data = await response.json(content_type=None)

            if isinstance(data, str):
                data = json.loads(data)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: {data}")

            print(data)
            result = {}
            for metal, info in data.get("data", {}).items():
                metal_info = {
                    "currency": info.get("currency", ""),
                    "name": info.get("name", ""),
                }

                item = info.get("results", [])
                if len(item) > 0:
                    item = item[0]
                    metal_info["bid"] = item.get("bid", "")
                    metal_info["mid"] = item.get("mid", "")
                    metal_info["high"] = item.get("high", "")
                    metal_info["low"] = item.get("low", "")
                    metal_info["originalTime"] = self._parse_time(item.get("originalTime", ""))
                    metal_info["unit"] = item.get("unit", "")

                result[metal] = metal_info

            return {"success": True, "data": {"base_currency": currency_code, "data": result}}

        except asyncio.TimeoutError:
            error_msg = f"Request timeout (timeout={self._timeout}s)"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except aiohttp.ClientError as e:
            error_msg = f"HTTP request error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Error occurred while getting metal price: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    def _parse_time(self, time_str: str) -> str:
        """Parse time string"""
        # "2025-04-25T17:00:00Z"
        # Convert to "2025-04-25 17:00:00"
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    import os
    import sys

    # 添加项目根目录到Python路径以允许直接运行此文件
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from external_api.data_sources.client import get_client

    async def main():
        client = get_client()
        result = await client.metal.get_metal_price(currency_code="USD")  # type: ignore
        print(result)
        print("\n")

    asyncio.run(main())
