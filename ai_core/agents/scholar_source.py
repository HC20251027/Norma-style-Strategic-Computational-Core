"""
学术数据源实现
"""

import asyncio
import logging
import math
from typing import Any, Dict, Optional

import aiohttp

from .base import BaseAPI

logger = logging.getLogger("scholar_source")


class ScholarSource(BaseAPI):
    """Academic data source

    Provide academic paper search capabilities based on Serper API.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the academic data source"""
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
        return "scholar"

    def get_api_info(self) -> Dict[str, Any]:
        """
        Get the basic information of the data source

        Returns:
            Dict[str, Any]: Contains name, description
        """
        return {"name": self.source_name, "description": "Scholar paper search, works like google scholar"}

    async def _fetch_scholar_page(
        self,
        query: str,
        page_size: int,
        page: int,
        start_year: Optional[str],
        end_year: Optional[str],
    ) -> Dict[str, Any]:
        """
        获取单页学术论文数据

        Args:
            query(str): 搜索关键词
            page_size(int): 每页数量
            page(int): 页码
            start_year(str): 开始年份
            end_year(str): 结束年份

        Returns:
            Dict[str, Any]: 单页搜索结果
        """
        payload = {"q": query, "page": page, "num": page_size}
        if start_year:
            payload["as_ylo"] = f"publication:{start_year}"
        if end_year:
            payload["as_yhi"] = f"publication:{end_year}"

        request_url = f"{self.proxy_url}/scholar"

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
                        "publicationInfo": item.get("publicationInfo"),
                        "year": item.get("year"),
                        "citedBy": item.get("citedBy"),
                        "pdfUrl": item.get("pdfUrl"),
                    }
                )
            return {"success": True, "data": results}
        except asyncio.TimeoutError:
            error_msg = f"Request timeout (timeout={self.timeout}s)"
            logger.error(f"_fetch_scholar_page error: page={page}, {error_msg}")
            return {"success": False, "error": error_msg}
        except aiohttp.ClientError as e:
            logger.error(f"_fetch_scholar_page error: page={page}, error={e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"_fetch_scholar_page error: page={page}, error={e}")
            return {"success": False, "error": str(e)}

    async def search_scholar(
        self,
        query: str,
        num_results: int = 10,
        start_year: Optional[str] = None,
        end_year: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for academic papers.

        Args:
            query(str): Search keywords.
            num_results(int): Number of results to return, default is 10, max is 500.
            start_year(str): Start year, YYYY, default is None.
            end_year(str): End year, YYYY, default is None.

        Returns:
            Dict[str, Any]: Search results, format:
                {
                    "success": True,
                    "data": {
                        "papers": [
                            {
                                "title": "...",
                                "snippet": "...",
                                "link": "...",
                                "publicationInfo": "...",
                                "year": "...",
                                "citedBy": "...",
                                "pdfUrl": "..."
                            }
                        ]
                    }
                }
        """

        # Example:
        #     >>> from external_api.data_sources.client import get_client
        #     >>> client = get_client()
        #     >>> print(f"Start scholar search")
        #     >>> result = await client.scholar.search_scholar(
        #     ...     query="machine learning",
        #     ...     num_results=100,
        #     ...     start_year="2020",
        #     ...     end_year="2023"
        #     ... )
        #     >>> if not result["success"]:
        #     ...     print(f"Search failed: {result['error']}")
        #     ... else:
        #     ...     print(f"Search succeeded, {len(result['data']['papers'])} results returned")
        # """
        try:
            # 限制最大结果数
            if num_results > 500:
                num_results = 500

            # 计算分页
            MAX_PAGE_SIZE = 20  # 最大每页数量,api有限制
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
                    self._fetch_scholar_page(
                        query=query,
                        page_size=current_page_size,
                        page=page,
                        start_year=start_year,
                        end_year=end_year,
                    )
                )

            # 等待所有请求完成
            results = await asyncio.gather(*tasks)

            # 合并结果
            all_papers = []
            has_error = False
            error_msgs = []
            successful_pages = 0

            for i, result in enumerate(results):
                page_num = i + 1
                if result["success"]:
                    page_results = result["data"]
                    all_papers.extend(page_results)
                    successful_pages += 1
                else:
                    has_error = True
                    error_msgs.append(f"Page {page_num}: {result['error']}")

            # 如果有部分失败，记录错误但仍返回成功获取的数据
            if has_error:
                logger.warning(f"Some scholar pages failed: {', '.join(error_msgs)}")

            # 限制返回数量
            all_papers = all_papers[:num_results]

            return {"success": True, "data": {"papers": all_papers}}
        except Exception as e:
            logger.error(f"search_scholar error: {e}")
            return {"success": False, "error": str(e)}
