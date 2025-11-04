"""
工具执行和结果处理工具
"""

import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import asdict
import uuid

from ..core.models import ToolExecutionContext, ToolExecutionResult
from ..core.result_handler import ResultProcessor, ResultFormat


logger = logging.getLogger(__name__)


class ToolRunner:
    """工具运行器"""
    
    def __init__(self):
        self.result_processor = ResultProcessor()
        self.execution_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5分钟缓存
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def run_tool(self, tool_id: str, parameters: Dict[str, Any], 
                      user_id: str = "user", format_type: str = "json") -> Dict[str, Any]:
        """运行工具"""
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(tool_id, parameters)
            
            # 检查缓存
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"返回缓存结果: {tool_id}")
                return cached_result
            
            # 创建执行上下文
            context = ToolExecutionContext(
                tool_id=tool_id,
                parameters=parameters,
                user_id=user_id
            )
            
            # 执行工具
            from ..core.executor import ToolExecutor
            executor = ToolExecutor()
            result = await executor.execute_tool(context)
            
            # 处理结果
            processed_result = await self.result_processor.process_result(
                result, 
                ResultFormat(format_type)
            )
            
            # 缓存结果
            self._cache_result(cache_key, processed_result)
            
            return processed_result
            
        except Exception as e:
            logger.error(f"工具运行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_id": tool_id
            }
    
    async def run_tool_async(self, tool_id: str, parameters: Dict[str, Any], 
                           user_id: str = "user") -> str:
        """异步运行工具，返回任务ID"""
        task_id = str(uuid.uuid4())
        
        # 创建任务
        task = asyncio.create_task(
            self.run_tool(tool_id, parameters, user_id)
        )
        
        self.running_tasks[task_id] = task
        
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """获取异步任务结果"""
        if task_id not in self.running_tasks:
            return None
        
        task = self.running_tasks[task_id]
        
        try:
            if timeout:
                result = await asyncio.wait_for(task, timeout=timeout)
            else:
                result = await task
            
            # 清理完成的任务
            del self.running_tasks[task_id]
            
            return result
            
        except asyncio.TimeoutError:
            return {"status": "timeout", "task_id": task_id}
        except Exception as e:
            logger.error(f"获取任务结果失败: {e}")
            return {"status": "error", "error": str(e), "task_id": task_id}
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            del self.running_tasks[task_id]
            return True
        return False
    
    def _generate_cache_key(self, tool_id: str, parameters: Dict[str, Any]) -> str:
        """生成缓存键"""
        param_str = json.dumps(parameters, sort_keys=True)
        return f"{tool_id}:{hashlib.md5(param_str.encode()).hexdigest()}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        if cache_key in self.execution_cache:
            cached = self.execution_cache[cache_key]
            # 检查是否过期
            if (datetime.now() - cached["timestamp"]).seconds < self.cache_ttl:
                return cached["result"]
            else:
                del self.execution_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """缓存结果"""
        self.execution_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now()
        }
        
        # 清理过期缓存
        self._cleanup_expired_cache()
    
    def _cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, cached in self.execution_cache.items():
            if (current_time - cached["timestamp"]).seconds >= self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.execution_cache[key]
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期缓存")


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results: Dict[str, Dict[str, Any]] = {}
    
    async def process_batch(self, tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """批量处理任务"""
        logger.info(f"开始批量处理 {len(tasks)} 个任务")
        
        # 创建任务
        async def process_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
            async with self.semaphore:
                tool_id = task.get("tool_id")
                parameters = task.get("parameters", {})
                task_id = task.get("id", str(uuid.uuid4()))
                
                try:
                    runner = ToolRunner()
                    result = await runner.run_tool(tool_id, parameters)
                    return {"task_id": task_id, "result": result}
                except Exception as e:
                    return {
                        "task_id": task_id, 
                        "result": {"success": False, "error": str(e)}
                    }
        
        # 并发执行
        batch_tasks = [process_single_task(task) for task in tasks]
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # 整理结果
        processed_results = {}
        for i, result in enumerate(results):
            task_id = tasks[i].get("id", f"task_{i}")
            
            if isinstance(result, Exception):
                processed_results[task_id] = {
                    "success": False,
                    "error": str(result)
                }
            else:
                processed_results[task_id] = result["result"]
        
        logger.info(f"批量处理完成，成功: {sum(1 for r in processed_results.values() if r.get('success'))}")
        
        return processed_results


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self):
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
    
    def register_workflow(self, workflow_id: str, workflow_def: Dict[str, Any]) -> bool:
        """注册工作流"""
        try:
            self.workflows[workflow_id] = workflow_def
            logger.info(f"工作流注册成功: {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"工作流注册失败: {e}")
            return False
    
    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any], 
                              user_id: str = "user") -> Dict[str, Any]:
        """执行工作流"""
        if workflow_id not in self.workflows:
            return {
                "success": False,
                "error": f"工作流不存在: {workflow_id}"
            }
        
        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        
        try:
            # 解析工作流定义
            steps = workflow.get("steps", [])
            variables = parameters.copy()
            results = {}
            
            # 按顺序执行步骤
            for step in steps:
                step_id = step.get("id")
                tool_id = step.get("tool")
                step_params = step.get("parameters", {})
                
                # 解析参数中的变量
                resolved_params = self._resolve_parameters(step_params, variables)
                
                # 执行工具
                runner = ToolRunner()
                result = await runner.run_tool(tool_id, resolved_params, user_id)
                
                results[step_id] = result
                
                # 更新变量
                if result.get("success"):
                    variables[f"step_{step_id}_result"] = result.get("data")
                
                # 检查是否继续
                if not result.get("success") and step.get("stop_on_error", True):
                    break
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "steps": steps,
                "results": results,
                "variables": variables
            }
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            return {
                "success": False,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "error": str(e)
            }
    
    def _resolve_parameters(self, parameters: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """解析参数中的变量"""
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                resolved[key] = variables.get(var_name, value)
            else:
                resolved[key] = value
        return resolved


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self):
        self.analyzers: Dict[str, Callable] = {}
    
    def register_analyzer(self, name: str, analyzer: Callable):
        """注册分析器"""
        self.analyzers[name] = analyzer
    
    async def analyze_results(self, results: Dict[str, Any], 
                             analysis_types: List[str]) -> Dict[str, Any]:
        """分析结果"""
        analysis_report = {
            "summary": {},
            "details": {},
            "recommendations": []
        }
        
        for analysis_type in analysis_types:
            if analysis_type in self.analyzers:
                try:
                    analyzer = self.analyzers[analysis_type]
                    if asyncio.iscoroutinefunction(analyzer):
                        analysis_result = await analyzer(results)
                    else:
                        analysis_result = analyzer(results)
                    
                    analysis_report["details"][analysis_type] = analysis_result
                    
                except Exception as e:
                    logger.error(f"分析器 {analysis_type} 执行失败: {e}")
                    analysis_report["details"][analysis_type] = {"error": str(e)}
        
        # 生成总结
        analysis_report["summary"] = self._generate_summary(results, analysis_report["details"])
        
        # 生成建议
        analysis_report["recommendations"] = self._generate_recommendations(analysis_report["details"])
        
        return analysis_report
    
    def _generate_summary(self, results: Dict[str, Any], analysis_details: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析总结"""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results.values() if r.get("success"))
        failed_tasks = total_tasks - successful_tasks
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (successful_tasks / max(total_tasks, 1)) * 100,
            "analysis_types": list(analysis_details.keys())
        }
    
    def _generate_recommendations(self, analysis_details: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于分析结果生成建议
        for analysis_type, details in analysis_details.items():
            if analysis_type == "performance" and details.get("avg_execution_time", 0) > 10:
                recommendations.append("平均执行时间较长，建议优化工具性能")
            
            elif analysis_type == "errors" and details.get("error_rate", 0) > 20:
                recommendations.append("错误率较高，建议检查工具配置和参数")
            
            elif analysis_type == "usage" and details.get("most_used_tool"):
                recommendations.append(f"最常用工具: {details['most_used_tool']}，可考虑优化该工具")
        
        return recommendations


# 预定义分析器
def performance_analyzer(results: Dict[str, Any]) -> Dict[str, Any]:
    """性能分析器"""
    execution_times = []
    for result in results.values():
        if result.get("success") and "execution_time" in result:
            execution_times.append(result["execution_time"])
    
    if not execution_times:
        return {"message": "无执行时间数据"}
    
    return {
        "avg_execution_time": sum(execution_times) / len(execution_times),
        "max_execution_time": max(execution_times),
        "min_execution_time": min(execution_times),
        "total_execution_time": sum(execution_times)
    }


def error_analyzer(results: Dict[str, Any]) -> Dict[str, Any]:
    """错误分析器"""
    errors = []
    tool_errors = {}
    
    for tool_id, result in results.items():
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            errors.append(error_msg)
            
            if tool_id not in tool_errors:
                tool_errors[tool_id] = []
            tool_errors[tool_id].append(error_msg)
    
    total_tasks = len(results)
    error_rate = (len(errors) / max(total_tasks, 1)) * 100
    
    return {
        "error_rate": error_rate,
        "total_errors": len(errors),
        "tool_errors": {k: len(v) for k, v in tool_errors.items()},
        "common_errors": list(set(errors))[:5]  # 前5个常见错误
    }


def usage_analyzer(results: Dict[str, Any]) -> Dict[str, Any]:
    """使用情况分析器"""
    tool_usage = {}
    total_executions = len(results)
    
    for result in results.values():
        # 这里应该从结果中提取工具ID
        # 简化实现
        tool_id = "unknown"
        tool_usage[tool_id] = tool_usage.get(tool_id, 0) + 1
    
    most_used_tool = max(tool_usage.items(), key=lambda x: x[1])[0] if tool_usage else None
    
    return {
        "total_executions": total_executions,
        "unique_tools": len(tool_usage),
        "tool_usage": tool_usage,
        "most_used_tool": most_used_tool
    }


# 全局实例
tool_runner = ToolRunner()
batch_processor = BatchProcessor()
workflow_executor = WorkflowExecutor()
result_analyzer = ResultAnalyzer()

# 注册预定义分析器
result_analyzer.register_analyzer("performance", performance_analyzer)
result_analyzer.register_analyzer("errors", error_analyzer)
result_analyzer.register_analyzer("usage", usage_analyzer)