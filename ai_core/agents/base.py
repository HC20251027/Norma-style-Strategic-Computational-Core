"""
数据源基础类定义

类的继承关系:
BaseApi (基类)
"""
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import os


EXCLUDE_METHODS = ['get_capabilities', 'get_api_info', 'source_name', 'get_source_info']

class BaseAPI(ABC):
    """
    数据源基类
    所有数据源都需要继承此类并实现相关方法
    """
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据源
        """
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        获取数据源名称

        Returns:
            str: 数据源名称
        """
        pass

    @abstractmethod
    def get_api_info(self) -> Dict[str, Any]:
        """
        获取数据源的基本信息

        Returns:
            Dict[str, Any]: 包含以下基本信息:
                - name: str, 数据源名称
                - description: str, 数据源描述
        """
        pass

    def get_capabilities(self) -> List[Dict[str, Any]]:
        """
        获取数据源所有能力的描述
        通过扫描实例方法及其文档字符串自动获取能力描述

        Returns:
            List[Dict[str, Any]]: 数据源提供的所有方法的描述列表
        """
        # 获取所有公开方法（不包括内置方法和私有方法）
        capabilities = []
        for attr_name in dir(self):
            if not attr_name.startswith('_'):  # 排除私有方法
                attr = getattr(self, attr_name)
                if callable(attr) and attr_name not in EXCLUDE_METHODS:
                    # 获取方法的文档字符串
                    doc = inspect.getdoc(attr)
                    if not doc:  # 跳过没有文档的方法
                        continue
                    if 'raise NotImplementedError' in inspect.getsource(attr):  # 跳过未实现的方法
                        continue
                    # 获取方法的签名
                    sig = inspect.signature(attr)
                    # 构建能力描述
                    capability = {
                        "name": attr_name,
                        "description": doc.split('\n\n')[0] if doc else "",  # 取第一段作为简短描述
                        "parameters": {
                            name: str(param.annotation).replace('typing.', '')
                            for name, param in sig.parameters.items()
                            if name != 'self'
                        },
                        "return_type": str(sig.return_annotation).replace('typing.', ''),
                        "doc": doc  # 完整的文档字符串
                    }
                    capabilities.append(capability)
        return capabilities