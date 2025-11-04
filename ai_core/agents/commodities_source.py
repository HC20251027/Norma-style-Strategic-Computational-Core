"""
Commodities data source implementation
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("commodities_source")


class CommoditiesSource(BaseAPI):
    """Commodity price data source"""

    def __init__(self, config: Dict[str, Any], proxy_url: Optional[str] = None):
        """Initialize Commodities price data source

        Args:
            config: Configuration dictionary containing API settings
            proxy_url: Optional proxy URL to override the default from config
        """
        self._timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        if proxy_url:
            self.proxy_url = proxy_url
        self._headers = {
            "X-Original-Host": config["commodities_base_url"],
            "X-Biz-Id": "matrix-agent",
            "X-Request-Timeout": str(config["timeout"] - 5),
        }

    @property
    def source_name(self) -> str:
        return "commodities"

    def get_api_info(self) -> Dict[str, Any]:
        """Get data source information"""
        return {
            "name": self.source_name,
            "description": "Commodity price data source, provides price information for commodities such as COCOA, COFFEE, CORN, OIL, SOYBEAN, SUGAR, WHEAT, etc.",
        }

    async def get_supported_commodities(self) -> Dict[str, Any]:
        """Get the list of supported commodities.
        This method is used to get the list of commodities that can be queried.

        Returns:
            Dict[str, Any]: Dictionary containing the list of supported commodities, e.g.
            {
                "success": True,
                "data": {
                    "commodities": [ # List of supported commodities
                        {
                            "commodity_code": "COCOA", # Commodity code, can be used to query price
                            "commodity_name": "Cocoa", # Commodity name
                            "commodity_weight_measurement": "Metric Ton (mt)" # Commodity unit
                        }, ...
                    ],
                    "currencies": [ # Supported currency types for price query
                        {
                            "currency_code": "USD", # Currency code, can be used to query price
                            "currency_name": "United States Dollar" # Currency name
                        }, ...
                    ]
                }
            }
        """
        #  Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> result = await client.commodities.get_supported_commodities()
        #     >>> if result["success"]:
        #     ...     print(f"Supported commodities: {result['data']['commodities']}")
        #     ...     print(f"Supported currencies: {result['data']['currencies']}")
        #     ... else:
        #     ...     print(f"Failed to get supported commodities: {result['error']}")
        # """
        try:
            request_url = f"{self.proxy_url}/v1/supported"

            # Send request using aiohttp
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(request_url, headers=self._headers, timeout=self._timeout) as response:
                    response.raise_for_status()

                    # Parse the response
                    data = await response.json(content_type=None)

            if isinstance(data, str):
                data = json.loads(data)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: {data}")

            if not data.get("success", False):
                raise ValueError(f"API response failed: {data}")

            return {
                "success": True,
                "data": {"commodities": data.get("supported_commodities", {}), "currencies": data.get("supported_currencies", {})},
            }

        except asyncio.TimeoutError:
            error_msg = f"Request timeout (timeout={self._timeout}s)"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except aiohttp.ClientError as e:
            error_msg = f"HTTP request error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Error occurred while getting supported commodities: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def get_commodities_price(
        self,
        commodity_code: str,
        currency_code: str,
    ) -> Dict[str, Any]:
        """
        Get commodity price.

        This method uses the commodities API to get commodity prices.

        Args:
            commodity_code(str): Commodity code, e.g. "COCOA,CORN,OIL", obtained from get_supported_commodities()
            currency_code(str): Currency code, e.g. "USD", obtained from get_supported_commodities()

        Returns:
            Dict[str, Any]: Dictionary containing the search results, e.g.
            {
                "success": True,               # Whether successful
                "data": {                      # If successful, contains the following fields
                    "base_currency": "USD", # Base currency code
                    "rates": {
                        "commodity_code": { # Queried commodity code
                            "open": 9270, # Opening price
                            "high": 9633, # Highest price
                            "low": 9201, # Lowest price
                            "prev": 9288, # Previous day's closing price
                            "current": 9590 # Current price
                        }
                    }
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> result = await client.commodities.get_commodities_price(
        #     ...     commodity_code="COCOA",
        #     ...     currency_code="USD"
        #     ... )
        #     >>> if result["success"]:
        #     ...     print(f"Successfully got commodity price")
        #     ... else:
        #     ...     print(f"Failed to get commodity price: {result['error']}")
        # """
        try:
            # Build query parameters
            params = {"symbols": commodity_code, "base": currency_code}

            request_url = f"{self.proxy_url}/v1/market-data"

            # Send request using aiohttp
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(request_url, headers=self._headers, params=params, timeout=self._timeout) as response:
                    response.raise_for_status()

                    # Parse the response
                    data = await response.json(content_type=None)

            if isinstance(data, str):
                data = json.loads(data)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: {data}")

            if not data.get("success", False):
                raise ValueError(f"API response failed: {data}")

            return {"success": True, "data": {"base_currency": data.get("base_currency", ""), "rates": data.get("rates", {})}}

        except asyncio.TimeoutError:
            error_msg = f"Request timeout (timeout={self._timeout}s)"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except aiohttp.ClientError as e:
            error_msg = f"HTTP request error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Error occurred while getting commodity price: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}


if __name__ == "__main__":
    from external_api.data_sources.client import get_client

    async def main():
        client = get_client()
        result1 = await client.commodities.get_supported_commodities()  # type: ignore
        print(result1)
        print("\n")
        result2 = await client.commodities.get_commodities_price(commodity_code="COCOA,CORN,OIL", currency_code="USD")  # type: ignore
        print(result2)

    asyncio.run(main())
