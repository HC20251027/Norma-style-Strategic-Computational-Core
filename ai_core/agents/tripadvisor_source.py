"""
TripAdvisor Officical API data source implementation
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from .base import BaseAPI

logger = logging.getLogger("tripadvisor_official_source")


class TripAdvisorSource(BaseAPI):
    """TripAdvisor official API data source"""

    def __init__(self, config: Dict[str, Any], proxy_url: Optional[str] = None):
        """Initialize TripAdvisor Official API data source"""
        self.timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        if proxy_url:
            self.proxy_url = proxy_url
        self.headers = {
            "X-Original-Host": config["tripadvisor_base_url"], 
            "X-Biz-Id":"matrix-agent",
            "X-Request-Timeout": str(config["timeout"]-5),
        }


    async def _make_api_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the Tripadvisor Content API"""
        url = f"{self.proxy_url}/api/v1/{endpoint}"
        
        if params is None:
            params = {}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()

    @property
    def source_name(self) -> str:
        return "tripadvisor"

    def get_api_info(self) -> Dict[str, Any]:
        """Get data source information"""
        return {
            "name": self.source_name,
            "description": "TripAdvisor official API data source, provides location info, reviews, and image search from TripAdvisor.",
        }

    async def search_locations(
        self,
        searchQuery: str,
        language: str = "en",
        category: Optional[str] = None,
        phone: Optional[str] = None,
        address: Optional[str] = None,
        latLong: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for locations (hotels, restaurants, attractions) on Tripadvisor

        Args:
            searchQuery(str): The text to search for
            language(str): Language code (default: 'en')
            category(str): Optional category filter ('hotels', 'attractions', 'restaurants', 'geos')
            phone(str): Optional phone number to search for
            address(str): Optional address to search for
            latLong(str): Optional latitude,longitude coordinates (e.g., '42.3455,-71.0983')

        Returns:
            Dict[str, Any]: Dictionary containing location info, e.g.
            {
                "success": True,               # Whether successful
                "data": [                      # If successful, contains the following fields
                    {
                        "location_id": "13189438", # Location ID
                        "name": "Hotel Xcaret Mexico", # Location name
                        "address_obj": { # Location address
                            "street1": "Carretera Federal Chetumal-Puerto Juarez, Av. Solidaridad 2-Kilometro 282", # Street
                            "city": "Playa del Carmen", # City
                            "state": "Quintana Roo", # State/Province
                            "country": "Mexico", # Country
                            "postalcode": "77710", # Postal code
                            "address_string": "Carretera Federal Chetumal-Puerto Juarez, Av. Solidaridad 2-Kilometro 282, Playa del Carmen 77710 Mexico" # Full address
                        }
                    },
                    ...
                ]
            }
        """
        params = {
            "searchQuery": searchQuery,
            "language": language,
        }

        if category:
            params["category"] = category
        if phone:
            params["phone"] = phone
        if address:
            params["address"] = address
        if latLong:
            params["latLong"] = latLong

        try:
            data = await self._make_api_request("location/search", params)
            if not data:
                return {"success": False, "error": "No data returned from Tripadvisor API"}
            if not data.get("data", None):
                return {"success": False, "error": "No data returned from Tripadvisor API"}
            return {"success": True, "data": data.get("data", [])}
        except Exception as e:
            logger.error(f"Error searching locations: {e}")
            return {"success": False, "error": str(e)}

    async def search_nearby_locations(
        self,
        latitude: float,
        longitude: float,
        language: str = "en",
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for locations near a specific latitude/longitude.

        Args:
            latitude(float): Latitude coordinate
            longitude(float): Longitude coordinate
            language(str): Language code (default: 'en')
            category(str): Optional category filter ('hotels', 'attractions', 'restaurants')

        Returns:
            Dict[str, Any]: Dictionary containing the search results
            {
                "success": True,               # Whether successful
                "data": [                      # If successful, contains the following fields
                    {
                        "location_id": "13189438", # Location ID
                        "name": "Hotel Xcaret Mexico", # Location name
                        "address_obj": { # Location address
                            "street1": "Carretera Federal Chetumal-Puerto Juarez, Av. Solidaridad 2-Kilometro 282", # Street
                            "city": "Playa del Carmen", # City
                            "state": "Quintana Roo", # State/Province
                            "country": "Mexico", # Country
                            "postalcode": "77710", # Postal code
                            "address_string": "Carretera Federal Chetumal-Puerto Juarez, Av. Solidaridad 2-Kilometro 282, Playa del Carmen 77710 Mexico" # Full address
                        }
                    },
                    ...
                ]
            }
        """
        params = {
            "latLong": f"{latitude},{longitude}",
            "language": language,
        }

        if category:
            params["category"] = category

        try:
            data = await self._make_api_request("location/nearby_search", params)
            if not data:
                return {"success": False, "error": "No data returned from Tripadvisor API"}

            if not data.get("data", None):
                return {"success": False, "error": "No data returned from Tripadvisor API"}

            return {"success": True, "data": data.get("data", [])}

        except Exception as e:
            logger.error(f"Error searching nearby locations: {e}")
            return {"success": False, "error": str(e)}

    async def get_location_details(
        self,
        locationId: int,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific location (hotel, restaurant, or attraction).

        Args:
            locationId(int): Tripadvisor location ID (can be string or integer)
            language(str): Language code (default: 'en')

        Returns:
            Dict[str, Any]: Dictionary containing detailed location info, e.g.
            {
                "success": True,               # Whether successful
                "data": {                      # If successful, contains the following fields
                    "location_id": "13189438", # Location ID
                    "name": "Hotel Xcaret Mexico", # Location name
                    "description": "...", # Location description
                    "web_url": "https://...", # Official website
                    "address_obj": {
                        "street1": "...", # Street
                        "city": "...", # City
                        "state": "...", # State/Province
                        "country": "...", # Country
                        "postalcode": "...", # Postal code
                        "address_string": "..." # Full address
                    },
                    "ancestors": [
                        {
                            "level": "...", # Level
                            "name": "...", # Name
                            "location_id": "..." # Location ID
                        },
                        ...
                    ],
                    "latitude": "...", # Latitude
                    "longitude": "...", # Longitude
                    "timezone": "...", # Timezone
                    "phone": "...", # Phone
                    "ranking_data": {
                        "geo_location_id": "150812", # Ranking region id
                        "ranking_string": "#27 of 392 hotels in Playa del Carmen", # Ranking info
                        "geo_location_name": "Playa del Carmen", # Ranking region name
                        "ranking_out_of": "392", # Total ranking
                        "ranking": "27" # Ranking position
                    },
                    "rating": "4.7", # Rating
                    "num_reviews": "14152", # Number of reviews
                    "review_rating_count": {
                        "1": "537", # Number of 1-star reviews, total 5 ratings
                    },
                    "subratings": { # Subrating details dict, contains multiple rating types
                        "0": {
                            "name": "rate_location", # Rating type
                            "localized_name": "Location", # Rating category name
                            "value": "4.8" # Rating value
                        },
                        ...
                    },
                    "photo_count": "20809", # Number of photos
                    "see_all_photos": "https://...", # See all photos link
                    "price_level": "$$$$", # Price level
                    "amenities": [], # Amenities list
                    "category": {
                        "name": "hotel", # Category name
                        "localized_name": "Hotel" # Localized category name
                    },
                    "subcategory": [
                        {
                            "name": "hotel", # Subcategory name
                            "localized_name": "Hotel" # Localized subcategory name
                        }
                    ],
                    "styles": [
                        "Trendy", # Style
                        "River View" # Style
                    ],
                    "neighborhood_info": [], # Neighborhood info
                    "trip_types": [ # Trip type data
                        {
                            "name": "business", # Trip type
                            "localized_name": "Business", # Localized trip type name
                            "value": "317" # Total trip type count
                        },
                        ...
                    ],
                    "awards": [] # Awards data
                }
            }
        """
        params = {
            "language": language,
        }

        # Convert locationId to string to ensure compatibility
        location_id_str = str(locationId)

        try:
            data = await self._make_api_request(f"location/{location_id_str}/details", params)
            if not data:
                return {"success": False, "error": "No data returned from Tripadvisor API"}

            return {"success": True, "data": self._parse_location_details(data)}
        except Exception as e:
            logger.error(f"Error getting location details: {e}")
            return {"success": False, "error": str(e)}

    async def get_location_reviews(
        self,
        locationId: int,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Get the most recent reviews for a specific location.

        Args:
            locationId(int): Tripadvisor location ID (can be string or integer)
            language(str): Language code (default: 'en')

        Returns:
            Dict[str, Any]: Dictionary containing review info, e.g.
            {
                "success": True,               # Whether successful
                "data": [                      # If successful, contains the following fields
                    {
                        "lang": "en", # Language code
                        "location_id": 13189438, # Location id
                        "published_date": "2025-04-22T21:05:13Z", # Review publish date
                        "rating": 5, # Rating
                        "helpful_votes": 0, # Helpful votes
                        "url": "https://...", # Review link
                        "text": "...", # Review content
                        "title": "...", # Review title
                        "trip_type": "Family", # Trip type
                        "travel_date": "2025-04-30", # Travel date
                        "user": { # Review user info
                            "username": "...", # Username
                            "avatar": {
                                "original": "https://...jpg" # User avatar
                            }
                        },
                        "subratings": { # Subrating details dict, contains multiple ratings
                            "0": {
                                "name": "RATE_VALUE", # Rating type
                                "value": 5, # Rating value
                                "localized_name": "Value" # Rating name
                            },
                            ...
                        },
                        "owner_response": { # Hotel reply
                            "id": 1004169956, # Reply id
                            "title": "Owner response", # Reply title
                            "text": "...", # Reply content
                            "lang": "en", # Reply language
                            "author": "Hotel Xcaret", # Reply author
                            "published_date": "2025-04-24T22:29:34Z" # Reply publish date
                        }
                    }
                ]
            }
        """
        params = {
            "language": language,
        }

        # Convert locationId to string to ensure compatibility
        location_id_str = str(locationId)

        try:
            data = await self._make_api_request(f"location/{location_id_str}/reviews", params)
            if not data:
                return {"success": False, "error": "No data returned from Tripadvisor API"}

            # 解析数据
            reviews = self._parse_reviews(data)
            return {"success": True, "data": reviews}
        except Exception as e:
            logger.error(f"Error getting location reviews: {e}")
            return {"success": False, "error": str(e)}

    async def get_location_photos(
        self,
        locationId: int,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Get high-quality photos for a specific location.

        Args:
            locationId(int): Tripadvisor location ID (can be string or integer)
            language(str): Language code (default: 'en')

        Returns:
            Dict[str, Any]: Dictionary containing photo info, e.g.
            {
                "success": True,               # Whether successful
                "data": [                      # If successful, contains the following fields
                    {
                        "id": 481190726, # Photo id
                        "is_blessed": False, # Is certified
                        "caption": "", # Photo caption
                        "published_date": "2021-02-26T00:50:50.206Z", # Photo publish date
                        "images": "https://...jpg" # Image url
                        "album": "Hotel & Grounds", # Photo album
                        "source": { # Photo source
                            "name": "Management", # Source name
                            "localized_name": "Management" # Localized source name
                        },
                        "user": { # Uploader
                            "username": "Management" # Username
                        }
                    },
                    ...
                ]
            }
        """
        params = {
            "language": language,
        }

        # Convert locationId to string to ensure compatibility
        location_id_str = str(locationId)

        try:
            data = await self._make_api_request(f"location/{location_id_str}/photos", params)
            if not data:
                return {"success": False, "error": "No data returned from Tripadvisor API"}

            return {"success": True, "data": self._parse_photos(data)}
        except Exception as e:
            logger.error(f"Error getting location photos: {e}")
            return {"success": False, "error": str(e)}

    def _parse_reviews(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse location review data"""
        reviews = []
        for review_data in data["data"]:
            subratings = {}
            for key, value in review_data.get("subratings", {}).items():
                subratings[key] = {
                    "name": value.get("name", ""),  # 评分类型
                    "value": value.get("value", ""),  # 评分值
                    "localized_name": value.get("localized_name", ""),  # 评分名称
                }

            review = {
                "lang": review_data.get("lang", "en"),  # 语言 code
                "location_id": review_data.get("location_id", ""),  # 地点 id
                "published_date": self._parse_date(review_data.get("published_date", "")),  # 评论发布时间
                "rating": review_data.get("rating", 0),  # 评分
                "helpful_votes": review_data.get("helpful_votes", 0),  # 有用投票数
                "url": review_data.get("url", ""),  # 评论链接
                "text": review_data.get("text", ""),  # 评论内容
                "title": review_data.get("title", ""),  # 评论标题
                "trip_type": review_data.get("trip_type", ""),  # 旅行类型
                "travel_date": review_data.get("travel_date", ""),  # 旅行日期
                "user": {  # 评论用户信息
                    "username": review_data.get("user", {}).get("username", ""),  # 用户名
                    "avatar": {
                        "original": review_data.get("user", {}).get("avatar", {}).get("original", "")  # 用户头像
                    },
                },
                "subratings": subratings,
                "owner_response": {  # 酒店回复
                    "id": review_data.get("owner_response", {}).get("id", ""),  # 回复 id
                    "title": review_data.get("owner_response", {}).get("title", ""),  # 回复标题
                    "text": review_data.get("owner_response", {}).get("text", ""),  # 回复内容
                    "lang": review_data.get("owner_response", {}).get("lang", ""),  # 回复语言
                    "author": review_data.get("owner_response", {}).get("author", ""),  # 回复作者名
                    "published_date": self._parse_date(review_data.get("owner_response", {}).get("published_date", "")),  # 回复发布时间
                },
            }
            reviews.append(review)

        return reviews

    def _parse_location_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse location detail data"""
        ancestors = []
        for ancestor in data.get("ancestors", []):
            ancestors.append(
                {
                    "level": ancestor.get("level", ""),  # 级别
                    "name": ancestor.get("name", ""),  # 名称
                    "location_id": ancestor.get("location_id", ""),  # 地点ID
                }
            )

        subratings = {}
        for key, value in data.get("subratings", {}).items():
            subratings[key] = {
                "name": value.get("name", ""),  # 评分类型
                "localized_name": value.get("localized_name", ""),  # 评分类别名称
                "value": value.get("value", ""),  # 评分值
            }

        trip_types = []
        for trip_type in data.get("trip_types", []):
            trip_types.append(
                {
                    "name": trip_type.get("name", ""),  # 旅行类型
                    "localized_name": trip_type.get("localized_name", ""),  # 旅行类型本地化名称
                    "value": trip_type.get("value", ""),  # 旅行类型总数
                }
            )

        subcategory = []
        for subcat in data.get("subcategory", []):
            subcategory.append(
                {
                    "name": subcat.get("name", ""),  # 子类别名称
                    "localized_name": subcat.get("localized_name", ""),  # 子类别本地化名称
                }
            )

        location_details = {  # 如果成功，包含以下字段
            "location_id": data.get("location_id", ""),  # 地点ID
            "name": data.get("name", ""),  # 地点名称
            "description": data.get("description", ""),  # 地点描述
            "web_url": data.get("web_url", ""),  # 地点官网链接
            "address_obj": {
                "street1": data.get("address_obj", {}).get("street1", ""),  # 街道位置
                "city": data.get("address_obj", {}).get("city", ""),  # 城市
                "state": data.get("address_obj", {}).get("state", ""),  # 州/省
                "country": data.get("address_obj", {}).get("country", ""),  # 国家
                "postalcode": data.get("address_obj", {}).get("postalcode", ""),  # 邮政编码
                "address_string": data.get("address_obj", {}).get("address_string", ""),  # 完整地址
            },
            "ancestors": ancestors,
            "latitude": data.get("latitude", ""),  # 纬度
            "longitude": data.get("longitude", ""),  # 经度
            "timezone": data.get("timezone", ""),  # 时区
            "phone": data.get("phone", ""),  # 电话
            "ranking_data": {
                "geo_location_id": data.get("ranking_data", {}).get("geo_location_id", ""),  # 排名地区 id
                "ranking_string": data.get("ranking_data", {}).get("ranking_string", ""),  # 排名信息
                "geo_location_name": data.get("ranking_data", {}).get("geo_location_name", ""),  # 排名地区名称
                "ranking_out_of": data.get("ranking_data", {}).get("ranking_out_of", ""),  # 排名总数
                "ranking": data.get("ranking_data", {}).get("ranking", ""),  # 排名位置
            },
            "rating": data.get("rating", ""),  # 评分
            "num_reviews": data.get("num_reviews", ""),  # 评论数
            "review_rating_count": data.get("review_rating_count", {}),
            "subratings": subratings,
            "photo_count": data.get("photo_count", ""),  # 照片数
            "see_all_photos": data.get("see_all_photos", ""),  # 查看所有照片链接
            "price_level": data.get("price_level", ""),  # 价格等级
            "amenities": data.get("amenities", []),  # 设施列表
            "category": {
                "name": data.get("category", {}).get("name", ""),  # 类别名称
                "localized_name": data.get("category", {}).get("localized_name", ""),  # 类别本地化名称
            },
            "subcategory": subcategory,
            "styles": data.get("styles", []),  # 风格
            "neighborhood_info": data.get("neighborhood_info", []),  # 邻里信息
            "trip_types": trip_types,
            "awards": data.get("awards", []),  # 奖项数据
        }

        return location_details

    def _parse_photos(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析地点照片数据"""
        photos = []
        for photo_data in data.get("data", []):
            photo = {
                "id": photo_data.get("id", ""),  # 照片 id
                "is_blessed": photo_data.get("is_blessed", False),  # 是否被认证
                "caption": photo_data.get("caption", ""),  # 照片描述
                "published_date": self._parse_date2(photo_data.get("published_date", "")),  # 照片发布时间
                "images": photo_data.get("images", {}).get("original", {}).get("url", ""),  # 图片 url
                "album": photo_data.get("album", ""),  # 照片所属相册
                "source": photo_data.get("source", {}),  # 照片来源
                "user": photo_data.get("user", {}),  # 照片上传者
            }
            photos.append(photo)

        return photos

    def _parse_date(self, date_str: str) -> str:
        """解析日期字符串"""
        # 新 API 日期格式：2025-04-24T22:29:34Z
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return date_str

    def _parse_date2(self, date_str: str) -> str:
        """解析日期字符串"""
        # 新 API 日期格式：2021-02-26T00:50:50.206Z
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return date_str


async def main():
    from external_api.data_sources.client import get_client

    client = get_client()
    print(await client.tripadvisor.search_locations(searchQuery="hotel", language="en"))  # type: ignore
    # print("\n")
    # print(await client.tripadvisor.search_nearby_locations(latitude=22.08, longitude=113.49, language="en"))
    # print("\n")
    # print(await client.tripadvisor.get_location_details(locationId=13189438, language="en"))
    # print("\n")
    # print(await client.tripadvisor.get_location_reviews(locationId=13189438, language="en"))
    # print("\n")
    # print(await client.tripadvisor.get_location_photos(locationId=13189438, language="en"))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
