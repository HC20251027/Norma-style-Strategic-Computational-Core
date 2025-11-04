"""
结果处理器
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import asyncio

from .models import ToolExecutionResult


logger = logging.getLogger(__name__)


class ResultFormat(Enum):
    """结果格式"""
    RAW = "raw"
    JSON = "json"
    TABLE = "table"
    CHART = "chart"
    TEXT = "text"
    HTML = "html"


class ResultProcessor:
    """结果处理器"""
    
    def __init__(self):
        self.formatters: Dict[ResultFormat, Callable] = {
            ResultFormat.RAW: self._format_raw,
            ResultFormat.JSON: self._format_json,
            ResultFormat.TABLE: self._format_table,
            ResultFormat.CHART: self._format_chart,
            ResultFormat.TEXT: self._format_text,
            ResultFormat.HTML: self._format_html
        }
        self.post_processors: List[Callable] = []
        self.result_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5分钟缓存
    
    async def process_result(
        self, 
        result: ToolExecutionResult, 
        format_type: ResultFormat = ResultFormat.JSON,
        post_process: bool = True
    ) -> Dict[str, Any]:
        """处理工具执行结果"""
        try:
            # 基础结果
            processed_result = {
                "success": result.success,
                "tool_id": result.tool_id,
                "request_id": result.request_id,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat(),
                "metadata": result.metadata
            }
            
            # 处理数据
            if result.success and result.data is not None:
                processed_result["data"] = await self._format_data(
                    result.data, format_type
                )
            else:
                processed_result["error"] = result.error
            
            # 后处理
            if post_process:
                processed_result = await self._apply_post_processors(processed_result)
            
            # 缓存结果
            cache_key = f"{result.tool_id}:{result.request_id}"
            self.result_cache[cache_key] = {
                "result": processed_result,
                "timestamp": datetime.now()
            }
            
            return processed_result
            
        except Exception as e:
            logger.error(f"结果处理失败: {e}")
            return {
                "success": False,
                "error": f"结果处理异常: {str(e)}",
                "tool_id": result.tool_id,
                "request_id": result.request_id
            }
    
    async def _format_data(self, data: Any, format_type: ResultFormat) -> Any:
        """格式化数据"""
        formatter = self.formatters.get(format_type, self._format_json)
        return await formatter(data) if asyncio.iscoroutinefunction(formatter) else formatter(data)
    
    async def _apply_post_processors(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """应用后处理器"""
        processed = result.copy()
        
        for processor in self.post_processors:
            try:
                if asyncio.iscoroutinefunction(processor):
                    processed = await processor(processed)
                else:
                    processed = processor(processed)
            except Exception as e:
                logger.warning(f"后处理器执行失败: {e}")
        
        return processed
    
    def _format_raw(self, data: Any) -> Any:
        """原始格式"""
        return data
    
    def _format_json(self, data: Any) -> Any:
        """JSON格式"""
        try:
            if isinstance(data, str):
                return json.loads(data)
            return data
        except (json.JSONDecodeError, TypeError):
            return {"raw_data": str(data)}
    
    def _format_table(self, data: Any) -> Dict[str, Any]:
        """表格格式"""
        if isinstance(data, list):
            if not data:
                return {"headers": [], "rows": []}
            
            # 如果是字典列表
            if isinstance(data[0], dict):
                headers = list(data[0].keys())
                rows = [[row.get(header, "") for header in headers] for row in data]
                return {"headers": headers, "rows": rows}
            
            # 如果是简单列表
            return {"headers": ["Value"], "rows": [[item] for item in data]}
        
        elif isinstance(data, dict):
            return {
                "headers": ["Key", "Value"],
                "rows": [[k, str(v)] for k, v in data.items()]
            }
        
        else:
            return {"headers": ["Value"], "rows": [[str(data)]]}
    
    def _format_chart(self, data: Any) -> Dict[str, Any]:
        """图表格式"""
        if isinstance(data, dict):
            # 假设数据是键值对，可以用于柱状图或饼图
            return {
                "type": "bar",
                "data": {
                    "labels": list(data.keys()),
                    "datasets": [{
                        "label": "Values",
                        "data": list(data.values())
                    }]
                }
            }
        
        elif isinstance(data, list) and data:
            if isinstance(data[0], dict) and "label" in data[0] and "value" in data[0]:
                return {
                    "type": "pie",
                    "data": {
                        "labels": [item["label"] for item in data],
                        "datasets": [{
                            "data": [item["value"] for item in data]
                        }]
                    }
                }
        
        # 默认返回原始数据
        return {"raw_data": data, "chart_type": "unknown"}
    
    def _format_text(self, data: Any) -> str:
        """文本格式"""
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)
    
    def _format_html(self, data: Any) -> str:
        """HTML格式"""
        if isinstance(data, dict):
            html = "<table border='1'>"
            html += "<tr><th>Key</th><th>Value</th></tr>"
            for k, v in data.items():
                html += f"<tr><td>{k}</td><td>{v}</td></tr>"
            html += "</table>"
            return html
        
        elif isinstance(data, list):
            html = "<table border='1'>"
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    html += "<tr>"
                    for k, v in item.items():
                        html += f"<td>{k}: {v}</td>"
                    html += "</tr>"
                else:
                    html += f"<tr><td>{item}</td></tr>"
            html += "</table>"
            return html
        
        return f"<p>{data}</p>"
    
    def add_post_processor(self, processor: Callable):
        """添加后处理器"""
        self.post_processors.append(processor)
    
    def remove_post_processor(self, processor: Callable):
        """移除后处理器"""
        if processor in self.post_processors:
            self.post_processors.remove(processor)
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        if cache_key in self.result_cache:
            cached = self.result_cache[cache_key]
            # 检查缓存是否过期
            if (datetime.now() - cached["timestamp"]).seconds < self.cache_ttl:
                return cached["result"]
            else:
                del self.result_cache[cache_key]
        return None
    
    def clear_cache(self):
        """清空缓存"""
        self.result_cache.clear()
        logger.info("结果缓存已清空")
    
    def cleanup_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, cached in self.result_cache.items():
            if (current_time - cached["timestamp"]).seconds >= self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.result_cache[key]
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计"""
        current_time = datetime.now()
        total_items = len(self.result_cache)
        expired_items = 0
        
        for cached in self.result_cache.values():
            if (current_time - cached["timestamp"]).seconds >= self.cache_ttl:
                expired_items += 1
        
        return {
            "total_items": total_items,
            "expired_items": expired_items,
            "valid_items": total_items - expired_items,
            "cache_ttl": self.cache_ttl
        }


class ResultValidator:
    """结果验证器"""
    
    def __init__(self):
        self.validators: Dict[str, Callable] = {}
    
    def add_validator(self, tool_id: str, validator: Callable):
        """添加验证器"""
        self.validators[tool_id] = validator
    
    async def validate_result(self, tool_id: str, result: ToolExecutionResult) -> tuple[bool, Optional[str]]:
        """验证结果"""
        validator = self.validators.get(tool_id)
        if not validator:
            return True, None  # 没有验证器，默认通过
        
        try:
            if asyncio.iscoroutinefunction(validator):
                is_valid, message = await validator(result)
            else:
                is_valid, message = validator(result)
            
            return is_valid, message
            
        except Exception as e:
            logger.error(f"结果验证失败: {e}")
            return False, f"验证过程出错: {str(e)}"


# 全局结果处理器实例
result_processor = ResultProcessor()
result_validator = ResultValidator()