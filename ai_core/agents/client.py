"""
统一的数据源访问客户端
"""

import importlib
import inspect
import logging
import os
import pkgutil
import threading
from enum import Enum
from pathlib import Path
from typing import Dict

from docstring_parser import parse

from .base import EXCLUDE_METHODS, BaseAPI

# 用于在shell中设置LLM_GATEWAY_BASE_URL环境变量
LLM_GATEWAY_BASE_URL_ENV_NAME = "LLM_GATEWAY_BASE_URL"

logger = logging.getLogger("data_sources_client")


def get_external_api_proxy_url() -> str:
    base_url = os.getenv(LLM_GATEWAY_BASE_URL_ENV_NAME) or "https://talkie-ali-virginia-prod-internal.xaminim.com"
    return f"{base_url}/llm/external-api"


config = {
    "name": "rapid_api",
    "twitter_base_url": "twitter154.p.rapidapi.com",
    "yahoo_base_url": "apidojo-yahoo-finance-v1.p.rapidapi.com",
    "booking_base_url": "booking-com15.p.rapidapi.com",
    "pinterest_base_url": "unofficial-pinterest-api.p.rapidapi.com",
    "tripadvisor_base_url": "api.content.tripadvisor.com",
    "commodities_base_url": "commodities-apised.p.rapidapi.com",
    "metal_base_url": "live-gold-prices.p.rapidapi.com",
    "serper_base_url": "google.serper.dev",
    "external_api_proxy_url": get_external_api_proxy_url(),
    "timeout": 60,
}


class ApiType(Enum):
    DATA_SOURCE = "data_source"
    FUNCTION = "function"


class ApiClient:
    """
    统一的数据源访问客户端
    负责管理和调用所有数据源

    使用单例模式，全局只初始化一次，线程安全
    """

    _exclude_sources = []

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:  # Double-check
                return
            self._sources: Dict[str, BaseAPI] = {}
            self._functions: Dict[str, BaseAPI] = {}
            self._load_data_sources()
            self._initialized = True

    def _load_data_sources(self):
        """
        动态加载所有可用的数据源
        通过扫描data_sources目录下的所有模块来加载数据源
        """
        current_dir = Path(__file__).parent
        for module_info in pkgutil.iter_modules([str(current_dir)]):
            type_dict = self._sources
            if module_info.name.endswith("_function"):
                type_dict = self._functions
            elif not module_info.name.endswith("_source"):
                continue

            try:
                module = importlib.import_module(f".{module_info.name}", package="external_api.data_sources")
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if (
                        isinstance(item, type)
                        and issubclass(item, BaseAPI)
                        and item != BaseAPI
                        and item.__name__ not in self._exclude_sources
                    ):
                        source = item(config)
                        type_dict[source.source_name] = source
            except Exception as e:
                logger.error(f"加载数据源模块 {module_info.name} 失败: {str(e)}\n")
                logger.exception(e)

    def get_function_desc(self, function_name: str) -> str:
        """
        Get a brief description and usage example of the specified function

        Args:
            function_name: str - function name

        Returns:
            str: Brief description and usage example of the function
        """
        return self._get_desc(ApiType.FUNCTION, function_name)

    def get_data_source_desc(self, source_name: str) -> str:
        """
        Get a brief description and usage example of the specified data source

        Args:
            source_name: str - data source name

        Returns:
            str: Readable description of the data source and its API
        """
        return self._get_desc(ApiType.DATA_SOURCE, source_name)

    def _get_desc(self, api_type: ApiType, api_name: str) -> str:
        """
        Get a brief description and usage example of the specified data source

        Args:
            api_type: ApiType - data source type
            api_name: str - data source name

        Returns:
            str: Readable description of the data source and its API
        """
        output_lines = ["# Available data sources (refer to the python code examples, write python code to call them)\n"]

        # Directly use the mapping value to get the data source instance
        if api_type == ApiType.DATA_SOURCE:
            api = self._sources.get(api_name)
        else:
            api = self._functions.get(api_name)

        if not api:
            return f"# {api_type.value} {api_name} does not exist"

        api_info = api.get_api_info()

        # Add data source title and description
        display_name = api_info.get("name", api_name)
        source_desc = api_info.get("description", "No description available")
        output_lines.extend([f"## {display_name}", f"{source_desc}\n"])

        # Get data source methods
        apis = []
        for method_name, method in inspect.getmembers(api.__class__, predicate=inspect.isfunction):
            # Skip internal methods
            if method_name.startswith("_") or method_name in EXCLUDE_METHODS:
                continue

            # Get method docstring
            doc = inspect.getdoc(method)
            if not doc:
                continue

            # Parse docstring
            docstring = parse(doc)

            # Prepare method description
            method_lines = [f"### {method_name}"]
            if docstring.short_description:
                method_lines.append(docstring.short_description + "\n")

            # Add parameter description
            if docstring.params:
                method_lines.append("**Parameters:**")
                for param in docstring.params:
                    param_desc = f"- `{param.arg_name}`"
                    if param.type_name:
                        param_desc += f": {param.type_name}"
                    if param.description:
                        param_desc += f" - {param.description}"
                    method_lines.append(param_desc)
                method_lines.append("")

            # Add return value description
            if docstring.returns:
                method_lines.append("**Returns:**")
                if docstring.returns.type_name:
                    method_lines.append(f"Type: `{docstring.returns.type_name}`")
                if docstring.returns.description:
                    method_lines.append("```")
                    method_lines.append(docstring.returns.description)
                    method_lines.append("```")
                method_lines.append("")

            # Add example
            if docstring.examples:
                method_lines.append("**Example:**")
                method_lines.append("```python")
                for example in docstring.examples:
                    if example.description:
                        # Directly add example code, no processing
                        method_lines.append(example.description.strip())
                method_lines.append("```")
                method_lines.append("")

            apis.extend(method_lines)

        # Merge all method descriptions
        if apis:
            output_lines.extend(apis)
        output_lines.append("---\n")

        return "\n".join(output_lines)

    def get_data_sources_basic_info(self) -> Dict[str, Dict[str, str]]:
        """
        Get basic information of all data sources, only including name and description

        Returns:
            Dict[str, Dict[str, str]]: Mapping of data source information, key is source_name, value is a dict containing display_name and description
        """
        result = {}

        for name, source in self._sources.items():
            # yahoo_finance和twitter 已通过 tool 实现，这里不展示
            if name in ["yahoo_finance", "twitter", "booking", "pinterest", "tripadvisor"]:
                continue

            source_info = source.get_api_info()

            # Get display name and description
            display_name = source_info.get("name", name)
            source_desc = source_info.get("description", "No description available")

            # Add to result dict
            result[name] = {"source_name": display_name, "description": source_desc}

        return result

    def get_all_function_desc(self) -> str:
        """
        获取所有数据源的所有方法的描述
        """
        result = []
        for function_name, function in self._functions.items():
            result.append(self.get_function_desc(function_name))
        return "\n".join(result)

    def __getattr__(self, name: str) -> BaseAPI:
        """
        Get data source instance by attribute access

        Args:
            name: data source name

        Returns:
            BaseDataSource: data source instance

        Raises:
            AttributeError: data source does not exist
        """
        if name not in self._sources:
            raise AttributeError(f"Data source {name} does not exist")
        return self._sources[name]


# 全局默认实例
_default_client = None
_client_lock = threading.Lock()


def get_client() -> ApiClient:
    """
    Get the default ApiClient instance

    Returns:
        ApiClient: Default ApiClient instance
    """
    global _default_client
    if _default_client is None:
        with _client_lock:
            if _default_client is None:  # Double-check
                _default_client = ApiClient()
    return _default_client
