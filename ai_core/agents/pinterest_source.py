"""
Pinterest data source implementation
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("pinterest_source")


class PinterestSource(BaseAPI):
    """Pinterest data source"""

    def __init__(self, config: Dict[str, Any], proxy_url: Optional[str] = None):
        """Initialize Pinterest data source"""
        self._timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        if proxy_url:
            self.proxy_url = proxy_url
        self._headers = {
            "X-Original-Host": config["pinterest_base_url"], 
            "X-Biz-Id":"matrix-agent",
            "X-Request-Timeout": str(config["timeout"]-5),
            }

    @property
    def source_name(self) -> str:
        return "pinterest"

    def get_api_info(self) -> Dict[str, Any]:
        """Get data source information"""
        return {"name": self.source_name, "description": "Pinterest data source, provides user and pin search features for Pinterest."}

    async def search_pins(
        self, keyword: str, num: int = 10, nextPageCursor: Optional[str] = None, sort: str = "relevance"
    ) -> Dict[str, Any]:
        """
        Search related pins.

        This method uses the Pinterest API to search for pins related to the given query.

        Args:
            keyword(str): Search keyword, e.g. "cats"
            num(int): Number of results per page, e.g. 10
            nextPageCursor(str): Pagination cursor for next page, default None for first page
            sort(str): Sort order, default "relevance", options: "relevance" or "recent"

        Returns:
            Dict[str, Any]: Dictionary containing pin search results, e.g.
            {
                "success": True,               # Whether successful
                "data": {                      # If successful, contains the following fields
                    "keyword": "cat",          # Search keyword
                    "count": 2,                # Number of pins returned
                    "pins": [                # Pin list
                        "id": "5559199536733192", # Pin id
                        "title": "cat", # Pin title
                        "description": "cat", # Pin description
                        "alt_text": "cat", # Image alt text
                        "auto_alt_text": "cat", # Image auto alt text
                        "images": { # Image info
                            "url": "https://xxx.jpg" # Image url
                        },
                        "videos": { # Video info
                            "has_video": Whether has video
                            "video_list": { # If has video, this field exists
                                "V_HLSV4": { # m3u8 format video, may not exist
                                    "url": "https://xxx.m3u8", # Video url
                                    "duration": 7000, # Video duration
                                },
                                "V_720P": { # 720p format video, may not exist
                                    "url": "https://xxx.mp4", # Video url
                                    "duration": 7000, # Video duration
                                }
                            }
                        },
                        "created_at": "2024-03-21 08:29:49",  # Created time
                        "likes": 635 # Number of likes
                        "pinner": { # Creator info
                            "id": "750412494069279813", # Creator id
                            "image_large_url": "https://xxxx.jpg", # Creator avatar url
                            "follower_count": 2379, # Follower count
                            "username": "Fursnpaws", # Creator username, can be used for search
                            "full_name": "FursnPaws | Dogs | Cats" # Creator display name
                        }
                    ],
                    "cursor": "cursor123"      # Next page cursor
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> result = await client.pinterest.search_pins(
        #     ...     keyword="cat",
        #     ...     num=10
        #     ... )
        #     >>> if result["success"]:
        #     ...     print(f"Found {result['data']['count']} pins")
        #     ... else:
        #     ...     print(f"Search failed: {result['error']}")
        # """
        try:
            # Build query parameters
            params = {"keyword": keyword, "num": num, "sort": sort}

            if nextPageCursor:
                params["nextPageCursor"] = nextPageCursor

            request_url = f"{self.proxy_url}/pinterest/pins/advance"

            # Send request using aiohttp
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(request_url, headers=self._headers, json=params, timeout=self._timeout) as response:
                    response.raise_for_status()
                    # Parse the response
                    data = await response.json(content_type=None)

            # The API returns a JSON string, need to parse it first
            if isinstance(data, str):
                data = json.loads(data)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: {data}")

            if "data" not in data:
                raise ValueError(f"API response missing data field: {data}")

            pins = self._parse_pins(data)

            return {"success": True, "data": {"keyword": keyword, "count": len(pins), "pins": pins, "cursor": data.get("nextPageCursor")}}

        except asyncio.TimeoutError:
            error_msg = f"Request timeout (timeout={self._timeout}s)"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except aiohttp.ClientError as e:
            error_msg = f"HTTP request error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Error occurred while searching pins: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def get_user_info(self, username: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information of a Pinterest user.

        Args:
            username (str): Pinterest username, not display name

        Returns:
            Dict[str, Any]: Dictionary containing user info, e.g.
            {
                "success": True,               # Whether successful
                "data": {                      # If successful, contains the following fields
                    "id": "750412494069279813", # User id
                    "full_name": "Display Name", # User display name
                    "username": "username", # Username, can be used for search
                    "image_url": "https://xxx.jpg", # User avatar url
                    "pin_count": 6459, # Number of pins published by user
                    "follower_count": 2385, # Number of followers
                    "last_pin_save_time": "2025-04-25 01:31:38", # Last pin publish time
                    "recent_pin_images": ["https://xxxx.jpg", ...] # Recent pin image urls
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Start to get user info of elonmusk")
        #     >>> result = await client.pinterest.get_user_info("elonmusk")
        #     >>> if result["success"]:
        #     ...     print(f"User display name: {result['data']['full_name']}")
        #     ...     print(f"Follower count: {result['data']['follower_count']}")
        #     ... else:
        #     ...     print(f"Failed to get user info: {result['error']}")
        # """
        try:
            # Build request URL
            request_url = f"{self.proxy_url}/pinterest/users/relevance"

            # Set request parameters
            params = {"keyword": username}

            # Send request using aiohttp
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(request_url, headers=self._headers, params=params, timeout=self._timeout) as response:
                    response.raise_for_status()
                    # Parse the response
                    data = await response.json(content_type=None)

            # Parse response data
            if isinstance(data, str):
                data = json.loads(data)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: {data}")

            # Build return data
            return {"success": True, "data": self._parse_user_info(data)}

        except asyncio.TimeoutError:
            error_msg = f"Request timeout (timeout={self._timeout}s)"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except aiohttp.ClientError as e:
            error_msg = f"HTTP request error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Error occurred while getting user info: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    def _format_date(self, date_str: Optional[str]) -> Optional[str]:
        """Format date string"""
        if not date_str:
            return None
        try:
            # New API date format example: "Thu Mar 13 18:08:35 +0000 2025"
            # New API date format example: "Tue, 04 Mar 2025 12:26:23 +0000",
            dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return date_str

    def _parse_pins(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        print(f"xwy-pins, {data}")
        print("-" * 100)
        print("\n")
        pins = []
        for pin_data in data.get("data", []):
            if not isinstance(pin_data, dict):
                logger.warning(f"Skip invalid pin data: {pin_data}")
                continue

            video = {"has_video": False}
            if pin_data.get("videos", None):
                V_HLSV4 = None
                if pin_data.get("videos", {}).get("video_list", {}).get("V_HLSV4", None):
                    V_HLSV4 = {
                        "url": pin_data.get("videos", {}).get("video_list", {}).get("V_HLSV4", {}).get("url", ""),
                        "duration": pin_data.get("videos", {}).get("video_list", {}).get("V_HLSV4", {}).get("duration", 0),
                    }

                V_720P = None
                if pin_data.get("videos", {}).get("video_list", {}).get("V_720P", None):
                    V_720P = {
                        "url": pin_data.get("videos", {}).get("video_list", {}).get("V_720P", {}).get("url", ""),
                        "duration": pin_data.get("videos", {}).get("video_list", {}).get("V_720P", {}).get("duration", 0),
                    }

                video: Dict[str, Any] = {"has_video": True}
                if V_HLSV4:
                    video["V_HLSV4"] = V_HLSV4
                if V_720P:
                    video["V_720P"] = V_720P

            image_url = pin_data.get("images", {}).get("original", {}).get("url", "")
            if len(image_url) <= 0:
                image_url = pin_data.get("images", {}).get("orig", {}).get("url", "")

            pin = {
                "id": pin_data.get("id", ""),
                "title": pin_data.get("title", ""),
                "description": pin_data.get("description", ""),
                "alt_text": pin_data.get("alt_text", ""),
                "auto_alt_text": pin_data.get("auto_alt_text", ""),
                "images": {"url": image_url},
                "videos": video,
                "created_at": "2024-03-21 08:29:49",  # 创建时间
                "likes": pin_data.get("reaction_counts", {}).get("1", 0),
                "pinner": {
                    "id": pin_data.get("pinner", {}).get("id", ""),
                    "image_url": pin_data.get("pinner", {}).get("image_large_url", ""),
                    "follower_count": pin_data.get("pinner", {}).get("follower_count", 0),
                    "username": pin_data.get("pinner", {}).get("username", ""),
                    "full_name": pin_data.get("pinner", {}).get("full_name", ""),
                },
            }
            pins.append(pin)
        return pins

    def _parse_user_info(self, resp: dict[str, Any]) -> dict[str, Any]:
        data = resp.get("data", [])
        print(f"xwy-user, {data}")
        print("-" * 100)
        print("\n")
        if len(data) <= 0:
            return {}

        data = data[0]

        recent_pin_images = []
        if data.get("recent_pin_images", None):
            key = list(data.get("recent_pin_images", {}).keys())[-1]
            for image_info in data.get("recent_pin_images", {}).get(key, {}):
                recent_pin_images.append(image_info.get("url", ""))

        return {
            "id": data.get("id", ""),  # User id
            "full_name": data.get("full_name", ""),  # User display name
            "username": data.get("username", ""),  # Username, can be used for search
            "image_url": data.get("image_large_url", ""),  # User avatar url
            "pin_count": data.get("pin_count", 0),  # Number of pins published by user
            "follower_count": data.get("follower_count", 0),  # Number of followers
            "last_pin_save_time": self._format_date(data.get("last_pin_save_time", "")),  # Last pin publish time
            "recent_pin_images": recent_pin_images,  # Recent pin image urls
        }


if __name__ == "__main__":
    from external_api.data_sources.client import get_client

    async def main():
        client = get_client()
        print(await client.pinterest.get_user_info("fursnpaws"))  # type: ignore
        print("\n")
        print(await client.pinterest.search_pins(keyword="cat", num=1))  # type: ignore

    asyncio.run(main())
