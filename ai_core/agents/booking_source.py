"""
Booking.com flight search data source implementation
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("booking_source")


class BookingSource(BaseAPI):
    """Booking.com data source"""

    def __init__(self, config: Dict[str, Any], proxy_url: Optional[str] = None):
        """Initialize Booking.com API data source"""

        self._timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        if proxy_url:
            self.proxy_url = proxy_url
        self.headers = {
            "X-Original-Host": config["booking_base_url"],
            "X-Biz-Id": "matrix-agent",
            "X-Request-Timeout": str(config["timeout"] - 5),
        }

    @property
    def source_name(self) -> str:
        """
        Get data source name

        Returns:
            str: Data source name
        """
        return "booking"

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get basic information about the data source

        Returns:
            Dict[str, Any]: Contains basic information like data source name and version
        """
        return {
            "name": self.source_name,
            "description": "Booking.com data source, providing flight search and hotel search services",
        }

    async def search_flights(
        self,
        from_code: str,
        to_code: str,
        depart_date: str,
        return_date: Optional[str] = None,
        stops: str = "none",
        page_no: int = 1,
        adults: int = 1,
        children: Optional[str] = None,
        sort: str = "BEST",
        cabin_class: str = "ECONOMY",
        currency_code: str = "USD",
    ) -> Dict[str, Any]:
        """
        Search for flights

        Args:
            from_code(str): Departure airport code, e.g.: PEK
            to_code(str): Destination airport code, e.g.: CAN
            depart_date(str): Departure date, format: YYYY-MM-DD
            return_date(Optional[str]): Return date, format: YYYY-MM-DD (optional)
            stops(str): Number of stops, options: none, 0, 1, 2
            page_no(int): Page number, default is 1
            adults(int): Number of adults, default is 1
            children(Optional[str]): Children's ages, comma separated, e.g.: 0,17 (optional)
            sort(str): Sort method, options: BEST, CHEAPEST, FASTEST
            cabin_class(str): Cabin class, options: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
            currency_code(str): Currency code, default USD

        Returns:
            Dict[str, Any]: Dictionary containing flight search results, e.g.
            {
                "success": True,                   # Whether successful
                "data": {                          # If successful, contains the following fields
                    "flights": [                   # Flight list
                        {
                            "stops": 0,            # Number of stops
                            "segments": [          # Segment information
                                {
                                    "flight_number": "CA1385",  # Flight number
                                    "from": "PEK", # Departure airport
                                    "to": "CAN",   # Arrival airport
                                    "departure": "2025-04-19T20:05:00",  # Departure time
                                    "arrival": "2025-04-19T23:10:00",     # Arrival time
                                    "total_time": 3.08  # Segment flight time
                                },
                                {
                                    "flight_number": "CA1386",
                                    "from": "CAN",
                                    "to": "PEK",
                                    "departure": "2025-04-26T06:25:00",
                                    "arrival": "2025-04-26T09:20:00",
                                    "total_time": 2.92  # Segment flight time
                                }
                            ],
                            "price": {             # Price information
                                "currency": "CNY", # Currency
                                "amount": 14272.26 # Total price
                            },
                            "total_time": 6.00  # Total flight time
                        }
                    ]
                }
            }
        """
        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Starting flight search")
        #     >>> result = await client.booking.search_flights(
        #     ...     from_code="PEK",
        #     ...     to_code="CAN",
        #     ...     depart_date="2025-04-19",
        #     ...     return_date="2025-04-26",
        #     ...     cabin_class="ECONOMY"
        #     ... )
        #     >>> if not result["success"]:
        #     ...     print(f"Search failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Search successful")
        # """
        try:
            # Build request parameters
            params = {
                "fromId": f"{from_code}.AIRPORT",
                "toId": f"{to_code}.AIRPORT",
                "departDate": depart_date,
                "stops": stops,
                "pageNo": page_no,
                "adults": adults,
                "sort": sort,
                "cabinClass": cabin_class,
                "currency_code": currency_code,
            }

            # Add optional parameters
            if return_date:
                params["returnDate"] = return_date
            if children:
                params["children"] = children

            request_url = f"{self.proxy_url}/api/v1/flights/searchFlights"

            logger.info("Starting flight search")

            # Send request
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        # Check response status
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

            # Check if API response has error
            if not data.get("status"):
                error_msg = data.get("message", "Unknown error")
                logger.error(f"API returned error: {error_msg}")
                return {"success": False, "error": error_msg}

            # 检查是否存在错误
            if not data.get("data", {}).get("flightOffers"):
                logger.error("No flight offers found")
                return {"success": True, "data": {"flights": []}}

            # Simplify response data structure
            simplified_flights = []
            for offer in data["data"]["flightOffers"]:
                legs_info = []
                stops_count = 0

                total_time = 0
                for segment in offer["segments"]:
                    # Get flight number and stop info
                    for leg in segment["legs"]:
                        flight_number = f"{leg['flightInfo']['carrierInfo']['marketingCarrier']}{leg['flightInfo']['flightNumber']}"
                        # Count stops
                        stops_count += len(leg.get("flightStops", []))

                        # Add segment info
                        legs_info.append(
                            {
                                "flight_number": flight_number,
                                "from": leg["departureAirport"]["code"],
                                "to": leg["arrivalAirport"]["code"],
                                "departure": leg["departureTime"],
                                "arrival": leg["arrivalTime"],
                                "total_time": self._format_duration(leg["totalTime"]),  # Segment flight time
                            }
                        )
                        total_time += leg["totalTime"]
                # Handle price
                price = offer["priceBreakdown"]["total"]
                total_amount = float(price["units"]) + float(price["nanos"]) / 1_000_000_000

                simplified_flights.append(
                    {
                        "stops": stops_count,
                        "segments": legs_info,
                        "total_time": self._format_duration(total_time),
                        "price": {"currency": price["currencyCode"], "amount": total_amount},
                    }
                )

            return {"success": True, "data": {"flights": simplified_flights}}

        except Exception as e:
            error_msg = f"Error occurred while searching flights: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def _search_hotel_destinations(self, query: str) -> Dict[str, Any]:
        """
        Search for hotel destinations

        Args:
            query(str): Search keyword, e.g.: shanghai

        Returns:
            Dict[str, Any]: Dictionary containing destination search results, e.g.
            {
                "success": True,                   # Whether successful
                "data": {                          # If successful, contains the following fields
                    "destinations": [              # Destination list
                        {
                            "dest_id": "-1924465", # Destination ID
                            "search_type": "city",   # Destination type
                            "name": "Shanghai",    # Destination name
                            "city_name": "Shanghai", # City name
                            "label": "Shanghai, Shanghai Area, China", # Full label
                            "longitude": 121.4763, # Longitude
                            "latitude": 31.229422, # Latitude
                            "country": "China"     # Country
                        }
                    ]
                }
            }
        """
        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Starting destination search")
        #     >>> result = await client.booking.search_destinations("shanghai")
        #     >>> if not result["success"]:
        #     ...     print(f"Search failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Search successful")
        # """
        try:
            # 构建请求参数
            params = {"query": query}

            request_url = f"{self.proxy_url}/api/v1/hotels/searchDestination"

            logger.info("Starting destination search")

            # 发送请求
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        # 检查响应状态
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

            # 检查API响应中是否有错误
            if not data.get("status"):
                error_msg = data.get("message", "Unknown error")
                logger.error(f"API returned error: {error_msg}")
                return {"success": False, "error": error_msg}

            # 简化响应数据结构
            simplified_destinations = []
            for dest in data["data"]:
                simplified_destinations.append(
                    {
                        "dest_id": dest["dest_id"],
                        "search_type": dest["search_type"],
                        "name": dest["name"],
                        "city_name": dest["city_name"],
                        "label": dest["label"],
                        "longitude": dest["longitude"],
                        "latitude": dest["latitude"],
                        "country": dest["country"],
                    }
                )

            return {"success": True, "data": {"destinations": simplified_destinations}}

        except Exception as e:
            error_msg = f"Error occurred while searching destinations: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def _search_hotels_by_destid(
        self,
        dest_id: str,
        search_type: str,
        arrival_date: str,
        departure_date: str,
        adults: int = 1,
        children_age: Optional[str] = None,
        room_qty: int = 1,
        page_number: int = 1,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        languagecode: str = "en-us",
        currency_code: str = "USD",
        sort_by: str = "bayesian_review_score",
        categories_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for hotels

        Args:
            dest_id(str): Destination ID, obtained from search_hotel_destinations
            search_type(str): Search type, obtained from search_hotel_destinations
            arrival_date(str): Check-in date, format: YYYY-MM-DD
            departure_date(str): Check-out date, format: YYYY-MM-DD
            adults(int): Number of adults, default is 1
            children_age(Optional[str]): Children's ages, comma separated, e.g.: 0,17
            room_qty(int): Number of rooms, default is 1
            page_number(int): Page number, default is 1
            price_min(Optional[float]): Minimum price, optional
            price_max(Optional[float]): Maximum price, optional
            languagecode(str): Language code, default en-us
            currency_code(str): Currency code, default USD
            sort_by(Optional[str]): Sort method, options:
                - upsort_bh: Entire homes & apartments first
                - popularity: Top picks for solo travellers
                - distance: Distance from city centre
                - class_descending: Property rating (5 to 0)
                - class_ascending: Property rating (0 to 5)
                - bayesian_review_score: Best reviewed first
                - price: Price (lowest first)
            categories_filter(Optional[str]): Star rating filter, options:
                - class::1: One star
                - class::2: Two stars
                - class::3: Three stars
                - class::4: Four stars
                - class::5: Five stars

        Returns:
            Dict[str, Any]: Dictionary containing hotel search results, e.g.
            {
                "success": True,                   # Whether successful
                "data": {                          # If successful, contains the following fields
                    "hotels": [                    # Hotel list
                        {
                            "hotel_id": "123456",  # Hotel ID
                            "name": "Atour Hotel Shanghai Bund", # Hotel name
                            "rating": 4,           # Star rating
                            "review_score": 8.5,   # Review score
                            "review_count": 570,   # Number of reviews
                            "location": {          # Location information
                                "latitude": 31.234571,
                                "longitude": 121.488426
                            },
                            "price": {             # Price information
                                "currency": "CNY", # Currency
                                "amount": 1758.78, # Total price
                                "price_per_night": 879.39 # Price per night
                            }
                        }
                    ]
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Starting hotel search")
        #     >>> result = await client.booking.search_hotels(
        #     ...     dest_id="-1924465",
        #     ...     search_type="CITY",
        #     ...     arrival_date="2025-04-19",
        #     ...     departure_date="2025-04-26"
        #     ... )
        #     >>> if not result["success"]:
        #     ...     print(f"Search failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Search successful")
        # """
        try:
            # 构建请求参数
            params = {
                "dest_id": dest_id,
                "search_type": search_type,
                "arrival_date": arrival_date,
                "departure_date": departure_date,
                "adults": adults,
                "room_qty": room_qty,
                "page_number": page_number,
                "languagecode": languagecode,
                "currency_code": currency_code,
                "units": "metric",
                "temperature_unit": "c",
            }

            # 添加可选参数
            if children_age:
                params["children_age"] = children_age
            if price_min is not None:
                params["price_min"] = price_min
            if price_max is not None:
                params["price_max"] = price_max
            if sort_by:
                params["sort_by"] = sort_by
            if categories_filter:
                params["categories_filter"] = categories_filter

            request_url = f"{self.proxy_url}/api/v1/hotels/searchHotels"

            logger.info("Starting hotel search")

            # 发送请求
            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        # 检查响应状态
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

            # 检查API响应中是否有错误
            if not data.get("status"):
                error_msg = data.get("message", "Unknown error")
                logger.error(f"API returned error: {error_msg}")
                return {"success": False, "error": error_msg}

            # 简化响应数据结构
            # 计算平均价格

            simplified_hotels = []
            for hotel in data["data"]["hotels"]:
                property_info = hotel["property"]
                avg_price = round(
                    property_info["priceBreakdown"]["grossPrice"]["value"]
                    / (datetime.strptime(departure_date, "%Y-%m-%d") - datetime.strptime(arrival_date, "%Y-%m-%d")).days,
                    2,
                )
                simplified_hotels.append(
                    {
                        "hotel_id": hotel["hotel_id"],
                        "name": property_info["name"],
                        "rating": property_info.get("accuratePropertyClass") or property_info.get("propertyClass"),
                        "review_score": property_info.get("reviewScore"),
                        "review_count": property_info.get("reviewCount"),
                        "location": {"latitude": property_info["latitude"], "longitude": property_info["longitude"]},
                        "price": {
                            "currency": property_info["priceBreakdown"]["grossPrice"]["currency"],
                            "amount": property_info["priceBreakdown"]["grossPrice"]["value"],
                            "price_per_night": avg_price,
                        },
                    }
                )

            return {"success": True, "data": {"hotels": simplified_hotels}}

        except Exception as e:
            error_msg = f"Error occurred while searching hotels: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def search_hotels_by_dest_name(
        self,
        dest_name: str,
        arrival_date: str,
        departure_date: str,
        adults: int = 1,
        children_age: Optional[str] = None,
        room_qty: int = 1,
        page_number: int = 1,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        languagecode: str = "en-us",
        currency_code: str = "USD",
        sort_by: str = "bayesian_review_score",
        categories_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for hotels by destination name

        Args:
            dest_name(str): Destination name, e.g.: shanghai
            arrival_date(str): Check-in date, format: YYYY-MM-DD
            departure_date(str): Check-out date, format: YYYY-MM-DD
            adults(int): Number of adults, default is 1
            children_age(Optional[str]): Children's ages, comma separated, e.g.: 0,17
            room_qty(int): Number of rooms, default is 1
            page_number(int): Page number, default is 1
            price_min(Optional[float]): Minimum price, optional
            price_max(Optional[float]): Maximum price, optional
            languagecode(str): Language code, default en-us
            currency_code(str): Currency code, default USD
            sort_by(Optional[str]): Sort method, options:
                - upsort_bh: Entire homes & apartments first
                - popularity: Top picks for solo travellers
                - distance: Distance from city centre
                - class_descending: Property rating (5 to 0)
                - class_ascending: Property rating (0 to 5)
                - bayesian_review_score: Best reviewed first
                - price: Price (lowest first)
            categories_filter(Optional[str]): Star rating filter, options:
                - class::1: One star, ..., class::5: Five stars
                - Multiple selection allowed, comma separated, e.g.: class::1,class::2

        Returns:
            Dict[str, Any]: Dictionary containing hotel search results, e.g.
            {
                "success": True,                   # Whether successful
                "data": {                          # If successful, contains the following fields
                    "destination": {               # Matched destination information
                        "name": "Shanghai",        # Destination name
                        "dest_id": "-1924465",     # Destination ID
                        "search_type": "city"      # Search type
                    },
                    "hotels": [                    # Hotel list
                        {
                            "hotel_id": "123456",  # Hotel ID
                            "name": "Atour Hotel Shanghai Bund", # Hotel name
                            "rating": 4,           # Star rating
                            "review_score": 8.5,   # Review score
                            "review_count": 570,   # Number of reviews
                            "location": {          # Location information
                                "latitude": 31.234571,
                                "longitude": 121.488426
                            },
                            "price": {             # Price information
                                "currency": "CNY", # Currency
                                "amount": 1758.78, # Total price
                                "price_per_night": 879.39 # Price per night
                            }
                        }
                    ]
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Starting hotel search")
        #     >>> result = await client.booking.search_hotels_by_dest_name(
        #     ...     dest_name="shanghai",
        #     ...     arrival_date="2025-04-19",
        #     ...     departure_date="2025-04-26"
        #     ... )
        #     >>> if not result["success"]:
        #     ...     print(f"Search failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Search successful")
        # """
        try:
            # 先搜索目的地信息
            dest_result = await self._search_hotel_destinations(dest_name)
            if not dest_result["success"]:
                return dest_result

            if not dest_result["data"]["destinations"]:
                return {"success": False, "error": f"No matching destination found: {dest_name}"}

            # 使用第一个匹配的目的地
            destination = dest_result["data"]["destinations"][0]
            dest_id = destination["dest_id"]
            search_type = destination["search_type"].upper()

            # 搜索酒店
            hotels_result = await self._search_hotels_by_destid(
                dest_id=dest_id,
                search_type=search_type,
                arrival_date=arrival_date,
                departure_date=departure_date,
                adults=adults,
                children_age=children_age,
                room_qty=room_qty,
                page_number=page_number,
                price_min=price_min,
                price_max=price_max,
                languagecode=languagecode,
                currency_code=currency_code,
                sort_by=sort_by,
                categories_filter=categories_filter,
            )

            if not hotels_result["success"]:
                return hotels_result

            # 在返回结果中添加目的地信息
            return {
                "success": True,
                "data": {
                    "destination": {
                        "name": destination["name"],
                        "dest_id": destination["dest_id"],
                        "search_type": destination["search_type"],
                    },
                    "hotels": hotels_result["data"]["hotels"],
                },
            }

        except Exception as e:
            error_msg = f"Error occurred while searching hotels: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    async def search_hotel_details(
        self,
        hotel_id: str,
        arrival_date: str,
        departure_date: str,
        adults: int = 1,
        children_age: Optional[str] = None,
        room_qty: int = 1,
        units: str = "metric",
        temperature_unit: str = "c",
        languagecode: str = "en-us",
        currency_code: str = "EUR",
    ) -> Dict[str, Any]:
        """
        Search for hotel details by hotel ID

        Args:
            hotel_id(str): Hotel ID
            arrival_date(str): Check-in date, format: YYYY-MM-DD
            departure_date(str): Check-out date, format: YYYY-MM-DD
            adults(int): Number of adults, default is 1
            children_age(Optional[str]): Children's ages, comma separated, e.g.: 0,17
            room_qty(int): Number of rooms, default is 1
            units(str): Units, default is metric
            temperature_unit(str): Temperature unit, default is c, options: c or f, where c = Celsius, f = Fahrenheit
            languagecode(str): Language code, default en-us
            currency_code(str): Currency code, default EUR

        Returns:
            Dict[str, Any]: Dictionary containing hotel details, e.g.
            {
                "success": True,                   # Whether successful
                "data": {                          # If successful, contains the following fields
                    "hotel_id": 191605,            # Hotel ID
                    "hotel_name": "Novotel Mumbai Juhu Beach", # Hotel name
                    "url": "https://...",          # Hotel URL
                    "review_nr": 2148,             # Number of reviews
                    "rating": 6.1,                 # Overall rating
                    "arrival_date": "2025-04-26",  # Check-in date
                    "departure_date": "2025-04-27", # Check-out date
                    "latitude": 19.1085017376187,  # Latitude
                    "longitude": 72.8243981301785, # Longitude
                    "address": "Juhu Beach, Maharastra", # Address
                    "city": "Mumbai",              # City name
                    "district": "Juhu Beach",      # District
                    "countrycode": "in",           # Country code
                    "country_trans": "India",      # Country name
                    "currency_code": "INR",        # Currency code
                    "zip": "400049",               # Postal code
                    "timezone": "Asia/Kolkata",    # Timezone
                    "rooms": {                     # Room information
                        "19160501": {
                            "photos": ["https://...", ...], # Room photos
                            "children_and_beds_text": {     # Children and beds information
                                "cribs_and_extra_beds": []  # Cribs and extra beds policy, may exist
                                "children_at_the_property": [] # Children policy, may exist
                                "allow_children": 1,        # Number of children allowed
                            },
                            "description": "...",           # Room description
                            "bed_configurations": [         # Bed configurations
                                {
                                    "name_with_count": "2 twin beds", # Bed count and name
                                    "description": "90–130 cm wide",  # Bed description
                                }, ...
                            ],
                        }, ...
                    }
                    "soldout": 0,                  # Whether sold out
                    "available_rooms": 7,          # Number of available rooms
                    "max_rooms_in_reservation": 7, # Maximum rooms in reservation
                    "average_room_size_for_ufi_m2": "14.07", # Average room size
                    "is_family_friendly": 0,       # Whether family friendly
                    "is_closed": 0,                # Whether closed
                    "is_cash_accepted_check_enabled": 1, # Whether cash is accepted
                    "hotel_include_breakfast": 1,  # Whether breakfast is included
                    "family_facilities": [...],    # Family facilities
                    "facilities": [...],           # Facilities list
                    "spoken_languages": [...],     # Available languages
                    "hotel_important_information_with_codes": [...], # Important notices
                }
            }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"开始请求酒店详情")
        #     >>> result = await client.booking.search_hotel_details(
        #     ...     hotel_id="191605",
        #     ...     arrival_date="2025-04-26",
        #     ...     departure_date="2025-04-27"
        #     ... )
        #     >>> if not result["success"]:
        #     ...     print(f"请求失败: {result['error']}")
        #     ... else:
        #     ...     print(f"请求成功")
        # """
        try:
            # 请求酒店详情
            # 构建请求参数
            params = {
                "hotel_id": hotel_id,
                "arrival_date": arrival_date,
                "departure_date": departure_date,
                "adults": adults,
                "room_qty": room_qty,
                "units": units,
                "temperature_unit": temperature_unit,
                "languagecode": languagecode,
                "currency_code": currency_code,
            }

            # 添加可选参数
            if children_age:
                params["children_age"] = children_age

            request_url = f"{self.proxy_url}/api/v1/hotels/getHotelDetails"

            try:
                async with aiohttp.ClientSession(trust_env=True) as session:
                    async with session.get(request_url, headers=self.headers, params=params, timeout=self._timeout) as response:
                        # 检查响应状态
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

            # 检查API响应中是否有错误
            if not data.get("status"):
                error_msg = data.get("message", "Unknown error")
                logger.error(f"API returned error: {error_msg}")
                return {"success": False, "error": error_msg}

            hotel_detail = self._parse_hotel_detail(data.get("data", {}))
            return {"success": True, "data": hotel_detail}
        except Exception as e:
            error_msg = f"Error occurred while searching hotel details: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return {"success": False, "error": error_msg}

    def _parse_hotel_detail(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解析酒店详情"""
        facilities = []
        for facility in data.get("facilities_block", {}).get("facilities", []):
            facility_name = facility.get("name", "")
            if len(facility_name) > 0:
                facilities.append(facility_name)

        hotel_important_information = []
        for item in data.get("hotel_important_information_with_codes", []):
            info = item.get("phrase", "")
            if len(info) > 0:
                hotel_important_information.append(info)

        rooms = {}
        for roomId, roomInfo in data.get("rooms", {}).items():
            room = {}
            photos_data = roomInfo.get("photos", [])
            photos = []
            for photo in photos_data:
                url = photo.get("url_max1280", "")
                if len(url) == 0:
                    url = photo.get("url_original", "")
                if len(url) > 0:
                    photos.append(url)

            children_and_beds_text = {}
            for key, value in roomInfo.get("children_and_beds_text", {}).items():
                if isinstance(value, list):
                    children_and_beds_text[key] = []
                    for item in value:
                        if len(item.get("text", "")) > 0:
                            children_and_beds_text[key].append(item.get("text", ""))
                elif isinstance(value, int):
                    children_and_beds_text[key] = value

            description = roomInfo.get("description", "")

            bed_configurations = []
            for bed_config in roomInfo.get("bed_configurations", []):
                for bed_type in bed_config.get("bed_types", []):
                    bed_name_cnt = bed_type.get("name_with_count", "")
                    bed_desc = bed_type.get("description", "")
                    bed_configurations.append({"name_with_count": bed_name_cnt, "description": bed_desc})

            room = {
                "photos": photos,
                "children_and_beds_text": children_and_beds_text,
                "description": description,
                "bed_configurations": bed_configurations,
            }
            rooms[roomId] = room

        hotel_detail = {
            "hotel_id": data.get("hotel_id", ""),  # 酒店 id
            "hotel_name": data.get("hotel_name", ""),  # 酒店名称
            "url": data.get("url", ""),  # 酒店url
            "review_nr": data.get("review_nr", ""),  # 评论数量
            "rating": data.get("raw_data", {}).get("reviewScore", ""),  # 综合评分
            "arrival_date": data.get("arrival_date", ""),  # 入住日期
            "departure_date": data.get("departure_date", ""),  # 离开日期
            "latitude": data.get("latitude", ""),  # 经度
            "longitude": data.get("longitude", ""),  # 纬度
            "address": data.get("address", ""),  # 地址
            "city": data.get("city", ""),  # 城市名
            "district": data.get("district", "") if data.get("district", "") != data.get("city", "") else "",  # 地址所在区
            "countrycode": data.get("countrycode", ""),  # 国家代码
            "country_trans": data.get("country_trans", ""),  # 国家名
            "currency_code": data.get("currency_code", ""),  # 货币代码
            "zip": data.get("zip", ""),  # 邮政编码
            "timezone": data.get("timezone", ""),  # 时区
            "soldout": data.get("soldout", ""),  # 是否售罄
            "available_rooms": data.get("available_rooms", ""),  # 可用房间数
            "max_rooms_in_reservation": data.get("max_rooms_in_reservation", ""),  # 最大预订房间数
            "average_room_size_for_ufi_m2": data.get("average_room_size_for_ufi_m2", ""),  # 平均房间大小
            "is_family_friendly": data.get("is_family_friendly", ""),  # 是否家庭友好
            "is_closed": data.get("is_closed", ""),  # 是否关门
            "is_cash_accepted_check_enabled": data.get("is_cash_accepted_check_enabled", ""),  # 是否接受现金
            "hotel_include_breakfast": data.get("hotel_include_breakfast", ""),  # 是否包含早餐
            "family_facilities": data.get("family_facilities", ""),  # 家庭设施
            "facilities": facilities,
            "spoken_languages": data.get("spoken_languages", []),  # 可用语言
            "hotel_important_information": hotel_important_information,
            "rooms": rooms,
        }
        return {"success": True, "data": hotel_detail}

    def _format_duration(self, seconds: int) -> str:
        """Convert seconds to hours and minutes format"""
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours} hours {mins} minutes"


if __name__ == "__main__":
    import json

    from external_api.data_sources.client import get_client

    async def main():
        client = get_client()
        result = await client.booking.search_hotel_details(  # type: ignore
            hotel_id="191605", arrival_date="2025-04-26", departure_date="2025-04-27"
        )
        print(json.dumps(result, indent=4))

    asyncio.run(main())
