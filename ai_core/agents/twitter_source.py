"""
Twitter data source implementation
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("twitter_source")


class TwitterSource(BaseAPI):
    """Twitter data source"""

    def __init__(self, config: Dict[str, Any], proxy_url: Optional[str] = None):
        """Initialize Twitter data source"""
        self._timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        if proxy_url:
            self.proxy_url = proxy_url
        self.headers = {
            "X-Original-Host": config["twitter_base_url"], 
            "X-Biz-Id":"matrix-agent",
            "X-Request-Timeout": str(config["timeout"]-5),
            }

    @property
    def source_name(self) -> str:
        return "twitter"

    def get_api_info(self) -> Dict[str, Any]:
        """Get data source information"""
        return {
            "name": self.source_name,
            "description": "Twitter data source, providing tweet search, user info retrieval, and user tweet list retrieval",
        }

    async def search_tweets(
        self,
        query: str,
        limit: int = 10,
        lang: Optional[str] = None,
        min_retweets: Optional[int] = None,
        min_likes: Optional[int] = None,
        min_replies: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for tweets.

        Args:
            query (str): Search keyword, e.g. "Tesla" or "#TSLA"
            limit (int): Maximum number of tweets to return, default is 10
            lang (Optional[str]): Language code, zh for Chinese, en for English, default is None
            min_retweets (Optional[int]): Minimum number of retweets, default is None
            min_likes (Optional[int]): Minimum number of likes, default is None
            min_replies (Optional[int]): Minimum number of replies, default is None
            start_date (Optional[str]): Start date, format: YYYY-MM-DD, default is None
            end_date (Optional[str]): End date, format: YYYY-MM-DD, default is None
            cursor (Optional[str]): Pagination cursor, used to get next page results, default is None for first page

        Returns:
            Dict[str, Any]: Dictionary containing tweet search results, e.g.
            {
                "success": True,               # Whether successful
                "data": {                      # If successful, contains the following fields
                    "query": "Tesla",          # Search keyword
                    "count": 2,                # Number of tweets returned
                    "tweets": [                # Tweet list
                        {
                            "id": "1234567890",           # Tweet ID
                            "created_at": "2024-03-21 08:29:49",  # Creation time
                            "text": "Tesla launch event was amazing!",     # Tweet content
                            "media_urls": ["https://..."],  # Media URL list
                            "video_urls": [],              # Video URL list
                            "author": {                    # Author information
                                "id": "987654321",         # Author ID
                                "name": "John Smith",      # Author name
                                "username": "johnsmith",   # Author username
                                "followers_count": 1000,   # Follower count
                                "is_verified": false,      # Whether verified
                                "is_blue_verified": false  # Whether blue verified
                            },
                            "public_metrics": {            # Public metrics
                                "retweet_count": 10,       # Retweet count
                                "reply_count": 5,          # Reply count
                                "like_count": 20,          # Like count
                                "quote_count": 2,          # Quote count
                                "view_count": 500,         # View count
                                "bookmark_count": 3        # Bookmark count
                            }
                        }
                    ],
                    "cursor": "cursor123"      # Next page cursor
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Starting tweet search")
        #     >>> result = await client.twitter.search_tweets(
        #     ...     query="Tesla",
        #     ...     limit=2,
        #     ...     lang="zh",
        #     ...     min_retweets=1,
        #     ...     min_likes=1,
        #     ...     min_replies=1,
        #     ...     start_date="2024-01-01",
        #     ...     end_date="2024-12-31"
        #     ... )
        #     >>> if result["success"]:
        #     ...     print(f"Found {result['data']['count']} tweets")
        #     ... else:
        #     ...     print(f"Search failed: {result['error']}")
        # """
        try:
            # 构建查询参数
            params = {
                "query": query,
                "section": "top",
                "limit": min(limit, 100),  # API限制最大100条
            }

            # 添加可选参数
            if min_retweets is not None:
                params["min_retweets"] = min_retweets
            if min_likes is not None:
                params["min_likes"] = min_likes
            if min_replies is not None:
                params["min_replies"] = min_replies
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            if cursor:
                params["continuation_token"] = cursor

            request_url = f"{self.proxy_url}/search/search"

            # 使用aiohttp发送异步请求
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                    response.raise_for_status()
                    # 解析响应
                    data = await response.json(content_type=None)

            # API返回的是JSON字符串，需要先解析
            if isinstance(data, str):
                data = json.loads(data)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: {data}")

            if "results" not in data:
                raise ValueError(f"Missing results field in API response: {data}")

            tweets = []
            for result in data["results"]:
                if not isinstance(result, dict):
                    logger.warning(f"Skipping invalid tweet data: {result}")
                    continue

                tweet = {
                    "id": str(result.get("tweet_id")),
                    "created_at": self._format_date(result.get("creation_date")),
                    "text": result.get("text", ""),
                    "media_urls": result.get("media_urls", []) if isinstance(result.get("media_urls"), list) else [],
                    "video_urls": result.get("video_urls", []) if isinstance(result.get("video_urls"), list) else [],
                    "author": {
                        "id": str(result.get("user", {}).get("user_id")),
                        "name": result.get("user", {}).get("name"),
                        "username": result.get("user", {}).get("username"),
                        "followers_count": result.get("user", {}).get("follower_count", 0),
                        "is_verified": result.get("user", {}).get("is_verified", False),
                        "is_blue_verified": result.get("user", {}).get("is_blue_verified", False),
                    },
                    "public_metrics": {
                        "retweet_count": result.get("retweet_count", 0),
                        "reply_count": result.get("reply_count", 0),
                        "like_count": result.get("favorite_count", 0),
                        "quote_count": result.get("quote_count", 0),
                        "view_count": result.get("views", 0),
                        "bookmark_count": result.get("bookmark_count", 0),
                    },
                }
                tweets.append(tweet)

            return {
                "success": True,
                "data": {"query": query, "count": len(tweets), "tweets": tweets, "cursor": data.get("continuation_token")},
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
            error_msg = f"Error occurred while searching tweets: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def get_user_info(self, username: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a Twitter user.

        Args:
            username (str): Twitter username without @ symbol
            user_id (Optional[str]): Twitter user ID, default is None, if provided user_id, username will be ignored

        Returns:
            Dict[str, Any]: Dictionary containing user information, e.g.
            {
                "success": True,               # Whether successful
                "data": {                      # If successful, contains the following fields
                    "id": "44196397",          # User ID
                    "username": "elonmusk",    # Username
                    "name": "Elon Musk",       # Display name
                    "created_at": "2009-06-02 20:12:29",  # Account creation time
                    "description": "Owner of X",  # Bio
                    "location": "Austin, TX",     # Location
                    "url": "https://x.com",       # Personal website
                    "profile_image_url": "https://...",   # Avatar URL
                    "profile_banner_url": "https://...",  # Banner image URL
                    "public_metrics": {           # Public metrics
                        "followers_count": 171500000,   # Follower count
                        "following_count": 1523,        # Following count
                        "tweet_count": 35420,           # Tweet count
                        "listed_count": 150200,         # Listed count
                        "like_count": 12000             # Like count
                    },
                    "verified": true,             # Whether verified
                    "blue_verified": true,        # Whether blue verified
                    "private": false,             # Whether private account
                    "bot": false                  # Whether bot account
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Start to get user info of elonmusk")
        #     >>> result = await client.twitter.get_user_info("elonmusk")
        #     >>> if result["success"]:
        #     ...     print(f"Username: {result['data']['name']}")
        #     ...     print(f"Followers: {result['data']['public_metrics']['followers_count']}")
        #     ... else:
        #     ...     print(f"Failed to get user info: {result['error']}")
        # """
        try:
            # 构建请求URL
            request_url = f"{self.proxy_url}/user/details"

            # 设置请求参数
            params = {"username": username}

            if user_id:
                params["user_id"] = user_id

            # 使用aiohttp发送异步请求
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                    response.raise_for_status()
                    # 解析响应
                    data = await response.json(content_type=None)

            # 解析响应数据
            if isinstance(data, str):
                data = json.loads(data)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: {data}")

            # 构建返回数据
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

    async def get_user_tweets(
        self, username: str, limit: int = 10, user_id: Optional[str] = None, include_replies: bool = False, include_pinned: bool = False
    ) -> Dict[str, Any]:
        """
        Get a list of tweets from a Twitter user.

        Args:
            username (str): Twitter username without @ symbol
            limit (int): Maximum number of tweets to return, default is 10
            user_id (Optional[str]): Twitter user ID, default is None, if provided user_id, username will be ignored
            include_replies (bool): Whether to include reply tweets, default is False
            include_pinned (bool): Whether to include pinned tweets, default is False

        Returns:
            Dict[str, Any]: Dictionary containing user tweet list, e.g.
            {
                "success": True,               # Whether successful
                "data": {                      # If successful, contains the following fields
                    "username": "elonmusk",    # Username
                    "count": 5,                # Number of tweets returned
                    "tweets": [                # Tweet list
                        {
                            "id": "1903001084357947836",  # Tweet ID
                            "created_at": "2024-03-21 08:29:49",  # Creation time
                            "text": "Many all-star engineers are taking major pay cuts...",  # Tweet content
                            "language": "en",              # Tweet language
                            "media_urls": ["https://..."],  # Media URL list
                            "video_urls": [],              # Video URL list
                            "public_metrics": {            # Public metrics
                                "retweet_count": 3848,     # Retweet count
                                "reply_count": 1511,       # Reply count
                                "like_count": 27328,       # Like count
                                "quote_count": 219,        # Quote count
                                "view_count": 2295512,     # View count
                                "bookmark_count": 0        # Bookmark count
                            },
                            "referenced_tweets": {         # Referenced tweets
                                "type": "retweet/quote/reply",
                                "id": "1902998745321468125",  # Referenced tweet ID
                                "text": "...",  # Referenced tweet content
                                ...  # Other fields
                            }
                        }
                    ],
                    "cursor": "cursor123"      # Next page cursor
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Start to get tweets of elonmusk")
        #     >>> result = await client.twitter.get_user_tweets(
        #     ...     username="elonmusk",
        #     ...     limit=5,
        #     ...     include_replies=True
        #     ... )
        #     >>> if result["success"]:
        #     ...     print(f"Retrieved {result['data']['count']} tweets")
        #     ...     for tweet in result["data"]["tweets"]:
        #     ...         print(f"- {tweet['text']}")
        #     ... else:
        #     ...     print(f"Failed to get tweets: {result['error']}")
        # """
        try:
            # 构建请求URL
            request_url = f"{self.proxy_url}/user/tweets"

            # 设置请求参数
            params = {
                "username": username,
                "limit": min(limit, 100),  # API限制最大100条
                "include_replies": str(include_replies).lower(),
                "include_pinned": str(include_pinned).lower(),
            }

            if user_id:
                params["user_id"] = user_id

            # 使用aiohttp发送异步请求
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                    response.raise_for_status()
                    # 解析响应
                    data = await response.json(content_type=None)

            # 解析响应数据
            if isinstance(data, str):
                data = json.loads(data)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid API response format: {data}")

            if "results" not in data:
                raise ValueError(f"Missing results field in API response: {data}")

            tweets = []
            for result in data["results"]:
                tweet = self._parse_tweet_with_ref(result)

                tweets.append(tweet)

            return {
                "success": True,
                "data": {"username": username, "count": len(tweets), "tweets": tweets, "cursor": data.get("continuation_token")},
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
            error_msg = f"Error occurred while getting user tweets: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    def _format_date(self, date_str: Optional[str]) -> Optional[str]:
        """Format date string"""
        if not date_str:
            return None
        try:
            # 新API的日期格式示例: "Thu Mar 13 18:08:35 +0000 2025"
            dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return date_str

    def _parse_user_info(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(data.get("user_id")),
            "username": data.get("username"),
            "name": data.get("name"),
            "created_at": self._format_date(data.get("creation_date")),
            "description": data.get("description"),
            "location": data.get("location"),
            "url": data.get("external_url"),
            "profile_image_url": data.get("profile_pic_url"),
            "profile_banner_url": data.get("profile_banner_url"),
            "public_metrics": {
                "followers_count": data.get("follower_count", 0),
                "following_count": data.get("following_count", 0),
                "tweet_count": data.get("number_of_tweets", 0),
                "listed_count": data.get("listed_count", 0),
                "like_count": data.get("favourites_count", 0),
            },
            "verified": data.get("is_verified", False),
            "blue_verified": data.get("is_blue_verified", False),
            "private": data.get("is_private", False),
            "bot": data.get("bot", False),
        }

    def _parse_tweet_without_ref(self, result: dict[str, Any]) -> dict[str, Any]:
        media_urls = []
        if result.get("media_url"):
            if isinstance(result["media_url"], list):
                media_urls.extend(result["media_url"])
            else:
                media_urls.append(result["media_url"])

        # 处理视频URL
        video_urls = []
        if result.get("video_url"):
            if isinstance(result["video_url"], list):
                video_urls.extend(result["video_url"])
            elif result["video_url"]:
                video_urls.append(result["video_url"])

        tweet = {
            "id": str(result.get("tweet_id")),
            "created_at": self._format_date(result.get("creation_date")),
            "text": result.get("text", ""),
            "language": result.get("language"),
            "media_urls": media_urls,
            "video_urls": video_urls,
            "public_metrics": {
                "retweet_count": result.get("retweet_count", 0),
                "reply_count": result.get("reply_count", 0),
                "like_count": result.get("favorite_count", 0),
                "quote_count": result.get("quote_count", 0),
                "view_count": result.get("views", 0),
                "bookmark_count": result.get("bookmark_count", 0),
            },
            "user": self._parse_user_info(result.get("user", {})),
        }

        return tweet

    def _parse_tweet_with_ref(self, result: dict[str, Any]) -> dict[str, Any]:
        """Parse tweet data"""

        tweet = self._parse_tweet_without_ref(result)

        # 处理引用推文
        referenced_tweets: dict[str, Any] = {}
        if result.get("in_reply_to_status_id"):
            referenced_tweets = {"type": "reply", "id": str(result.get("in_reply_to_status_id", ""))}
        elif result.get("retweet_tweet_id") and result.get("retweet_status"):
            retweet = result.get("retweet_status", {})
            referenced_tweets = {"type": "retweet", **self._parse_tweet_without_ref(retweet)}
            if retweet.get("quoted_status"):
                quoted = retweet.get("quoted_status", {})
                referenced_tweets["quoted_status"] = {"type": "quote", **self._parse_tweet_without_ref(quoted)}
        elif result.get("quoted_status_id") and result.get("quoted_status"):
            quoted = result.get("quoted_status", {})
            referenced_tweets = {"type": "quote", **self._parse_tweet_without_ref(quoted)}

        if referenced_tweets:
            tweet["referenced_tweets"] = referenced_tweets

        return tweet
