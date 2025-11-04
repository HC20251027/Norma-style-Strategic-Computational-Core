"""
专利数据源实现
"""

import asyncio
import logging
import math
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("patents_source")


class PatentSource(BaseAPI):
    """Patent data source"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the patent data source"""
        self.timeout = config["timeout"]
        self.proxy_url = config["external_api_proxy_url"]
        self.headers = {
            "X-Original-Host": config["serper_base_url"],
            "X-Biz-Id": "matrix-agent",
            "X-Request-Timeout": str(config["timeout"] - 5),
        }

    @property
    def source_name(self) -> str:
        """
        Get the name of the data source

        Returns:
            str: The name of the data source
        """
        return "patent"

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get the basic information of the data source

        Returns:
            Dict[str, Any]: Contains name, description
        """
        return {"name": self.source_name, "description": "Patent search, works like google patents"}

    async def _fetch_patents_page(
        self,
        query: str,
        assignee: Optional[str],
        page_size: int,
        page: int,
        start_time: Optional[str],
        end_time: Optional[str],
    ) -> Dict[str, Any]:
        """
        获取单页专利数据

        Args:
            query(str): 搜索关键词
            assignee(str): 专利所有者
            page_size(int): 每页数量
            page(int): 页码
            start_time(str): 开始时间
            end_time(str): 结束时间

        Returns:
            Dict[str, Any]: 单页搜索结果
        """
        payload = {"q": query, "num": page_size, "page": page}
        if assignee:
            payload["assignee"] = assignee
        if start_time:
            payload["after"] = f"publication:{start_time}"
        if end_time:
            payload["before"] = f"publication:{end_time}"

        request_url = f"{self.proxy_url}/patents"

        try:
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.post(request_url, headers=self.headers, json=payload, timeout=self.timeout) as response:
                    response.raise_for_status()
                    data = await response.json()

            organic = data.get("organic", [])
            results = []
            for item in organic:
                results.append(
                    {
                        "title": item.get("title"),
                        "snippet": item.get("snippet"),
                        "link": item.get("link"),
                        "priorityDate": item.get("priorityDate"),
                        "filingDate": item.get("filingDate"),
                        "grantDate": item.get("grantDate"),
                        "inventor": item.get("inventor"),
                        "assignee": item.get("assignee"),
                        "publicationNumber": item.get("publicationNumber"),
                        "pdfUrl": item.get("pdfUrl"),
                    }
                )
            return {"success": True, "data": results}
        except Exception as e:
            logger.error(f"_fetch_patents_page error: page={page}, error={e}")
            return {"success": False, "error": str(e)}

    async def search_patents(
        self,
        query: str,
        assignee: Optional[str] = None,
        num_results: int = 10,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for patents.

        Args:
            assignee(str): The assignee of the patents, e.g. "Apple Inc.".
            query(str): Search keywords. up to 5.
            num_results(int): Number of results to return, default is 10, max is 500
            start_time(str): Start date YYYYMMDD, optional.
            end_time(str): End date YYYYMMDD, optional.

        Returns:
            Dict[str, Any]: Search results, format:
                {
                    "success": True,
                    "data": {
                        "patents": [
                            {
                                "title": "...",
                                "snippet": "...",
                                "link": "...",
                                "priorityDate": "...",
                                "filingDate": "...",
                                "grantDate": "...",
                                "inventor": "...",
                                "assignee": "...",
                                "publicationNumber": "...",
                                "pdfUrl": "..."
                            }
                        ]
                    }
                }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Start patent search")
        #     >>> result = await client.patent.search_patents(
        #     ...     assignee="Apple Inc.",
        #     ...     query="machine learning",
        #     ...     num_results=100,
        #     ...     start_time="20200101",
        #     ...     end_time="20231231"
        #     ... )
        #     >>> if not result["success"]:
        #     ...     print(f"Search failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Search succeeded, {len(result['data']['patents'])} results returned")
        # """
        try:
            # 关键词裁剪
            keywords = query.split(" ")
            if len(keywords) > 5:
                query = " ".join(keywords[:5])

            # 限制最大结果数
            if num_results > 500:
                num_results = 500

            # 计算分页
            MAX_PAGE_SIZE = 50
            page_size = min(num_results, MAX_PAGE_SIZE)
            total_pages = math.ceil(num_results / page_size)

            # 并发请求所有页面
            tasks = []
            for page in range(1, total_pages + 1):
                # 最后一页可能需要调整数量
                if page == total_pages and num_results % page_size != 0:
                    current_page_size = num_results % page_size
                else:
                    current_page_size = page_size

                tasks.append(
                    self._fetch_patents_page(
                        query=query,
                        assignee=assignee,
                        page_size=current_page_size,
                        page=page,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )

            # 等待所有请求完成
            results = await asyncio.gather(*tasks)

            # 合并结果
            all_patents = []
            has_error = False
            error_msgs = []

            for result in results:
                if result["success"]:
                    all_patents.extend(result["data"])
                else:
                    has_error = True
                    error_msgs.append(result["error"])

            # 如果有部分失败，记录错误但仍返回成功获取的数据
            if has_error:
                logger.warning(f"Some patent pages failed: {', '.join(error_msgs)}")

            # 限制返回数量
            all_patents = all_patents[:num_results]

            return {"success": True, "data": {"patents": all_patents}}
        except Exception as e:
            logger.error(f"search_patents error: {e}")
            return {"success": False, "error": str(e)}
