"""
错误处理器

负责工具调用的错误处理、重试机制、故障恢复等
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import traceback

from ..models import ToolcallContext, ExecutionPlan


logger = logging.getLogger(__name__)


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 错误统计
        self.error_statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_errors": 0,
            "error_types": defaultdict(int),
            "last_error_time": None,
            "retry_count": 0,
            "recovery_count": 0
        })
        
        # 错误历史
        self.error_history: deque = deque(maxlen=1000)
        
        # 故障恢复策略
        self.recovery_strategies: Dict[str, Callable] = {
            "timeout": self._handle_timeout_error,
            "connection": self._handle_connection_error,
            "resource": self._handle_resource_error,
            "permission": self._handle_permission_error,
            "validation": self._handle_validation_error,
            "unknown": self._handle_unknown_error
        }
        
        # 熔断器状态
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    async def handle_execution_error(
        self, 
        tool_id: str, 
        error: Exception, 
        context: ToolcallContext,
        retry_func: Callable,
        execution_count: int = 0
    ) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """处理执行错误"""
        error_info = self._analyze_error(error)
        
        # 记录错误
        await self._record_error(tool_id, error_info, context, execution_count)
        
        # 检查是否应该重试
        should_retry, retry_reason = await self._should_retry(tool_id, error_info, execution_count)
        
        if not should_retry:
            # 尝试故障恢复
            recovery_result = await self._attempt_recovery(tool_id, error_info, context, retry_func)
            if recovery_result:
                return True, recovery_result, {"recovery": True, "reason": retry_reason}
            
            # 熔断处理
            await self._handle_circuit_breaker(tool_id, error_info)
            
            return False, None, {
                "error": str(error),
                "error_type": error_info["type"],
                "retry_exhausted": True,
                "reason": retry_reason
            }
        
        # 执行重试
        try:
            logger.info(f"重试工具 {tool_id}，第 {execution_count + 1} 次尝试")
            
            # 等待重试延迟
            if execution_count > 0:
                delay = self._calculate_retry_delay(execution_count, error_info)
                await asyncio.sleep(delay)
            
            # 执行重试
            result = await retry_func()
            
            # 重试成功，记录恢复
            await self._record_recovery(tool_id, execution_count)
            
            return True, result, {
                "retry": True,
                "attempt": execution_count + 1,
                "delay": delay if execution_count > 0 else 0
            }
            
        except Exception as retry_error:
            logger.error(f"重试失败: {retry_error}")
            
            # 递归处理重试错误
            return await self.handle_execution_error(
                tool_id, retry_error, context, retry_func, execution_count + 1
            )
    
    async def handle_plan_execution_error(
        self, 
        plan: ExecutionPlan, 
        failed_tools: List[str], 
        context: ToolcallContext,
        executor_func: Callable
    ) -> Tuple[bool, ExecutionPlan]:
        """处理计划执行错误"""
        logger.info(f"处理计划执行错误，失败工具: {failed_tools}")
        
        # 分析失败原因
        failure_analysis = await self._analyze_plan_failures(plan, failed_tools, context)
        
        # 生成恢复计划
        recovery_plan = await self._generate_recovery_plan(plan, failure_analysis)
        
        # 执行恢复计划
        recovery_success = await self._execute_recovery_plan(recovery_plan, context, executor_func)
        
        return recovery_success, recovery_plan
    
    async def _should_retry(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        execution_count: int
    ) -> Tuple[bool, str]:
        """判断是否应该重试"""
        # 检查重试次数
        if execution_count >= self.max_retries:
            return False, f"已达到最大重试次数 ({self.max_retries})"
        
        # 检查错误类型
        error_type = error_info["type"]
        
        # 可重试的错误类型
        retryable_errors = {"timeout", "connection", "resource", "rate_limit"}
        
        if error_type not in retryable_errors:
            return False, f"错误类型 {error_type} 不可重试"
        
        # 检查熔断器状态
        if self._is_circuit_breaker_open(tool_id):
            return False, "熔断器开启，暂停重试"
        
        # 检查错误频率
        if self._is_error_rate_too_high(tool_id):
            return False, "错误频率过高，暂停重试"
        
        return True, "可以重试"
    
    async def _attempt_recovery(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        context: ToolcallContext,
        retry_func: Callable
    ) -> Optional[Any]:
        """尝试故障恢复"""
        error_type = error_info["type"]
        
        recovery_strategy = self.recovery_strategies.get(
            error_type, 
            self.recovery_strategies["unknown"]
        )
        
        try:
            recovery_result = await recovery_strategy(tool_id, error_info, context, retry_func)
            return recovery_result
        except Exception as e:
            logger.error(f"故障恢复失败: {e}")
            return None
    
    async def _handle_timeout_error(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        context: ToolcallContext,
        retry_func: Callable
    ) -> Optional[Any]:
        """处理超时错误"""
        logger.info(f"处理超时错误: {tool_id}")
        
        # 延长超时时间重试
        original_timeout = context.metadata.get("timeout", 30)
        context.metadata["timeout"] = original_timeout * 2
        
        try:
            return await retry_func()
        finally:
            context.metadata["timeout"] = original_timeout
    
    async def _handle_connection_error(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        context: ToolcallContext,
        retry_func: Callable
    ) -> Optional[Any]:
        """处理连接错误"""
        logger.info(f"处理连接错误: {tool_id}")
        
        # 等待网络恢复
        await asyncio.sleep(5)
        
        # 重试连接
        return await retry_func()
    
    async def _handle_resource_error(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        context: ToolcallContext,
        retry_func: Callable
    ) -> Optional[Any]:
        """处理资源错误"""
        logger.info(f"处理资源错误: {tool_id}")
        
        # 释放资源后重试
        await self._cleanup_resources()
        await asyncio.sleep(2)
        
        return await retry_func()
    
    async def _handle_permission_error(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        context: ToolcallContext,
        retry_func: Callable
    ) -> Optional[Any]:
        """处理权限错误"""
        logger.info(f"处理权限错误: {tool_id}")
        
        # 权限错误通常不可恢复，返回None
        return None
    
    async def _handle_validation_error(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        context: ToolcallContext,
        retry_func: Callable
    ) -> Optional[Any]:
        """处理验证错误"""
        logger.info(f"处理验证错误: {tool_id}")
        
        # 验证错误通常不可重试，返回None
        return None
    
    async def _handle_unknown_error(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        context: ToolcallContext,
        retry_func: Callable
    ) -> Optional[Any]:
        """处理未知错误"""
        logger.info(f"处理未知错误: {tool_id}")
        
        # 未知错误，尝试简单重试
        await asyncio.sleep(1)
        return await retry_func()
    
    def _analyze_error(self, error: Exception) -> Dict[str, Any]:
        """分析错误信息"""
        error_type = type(error).__name__.lower()
        
        # 分类错误类型
        if "timeout" in str(error).lower():
            categorized_type = "timeout"
        elif "connection" in str(error).lower() or "network" in str(error).lower():
            categorized_type = "connection"
        elif "resource" in str(error).lower() or "memory" in str(error).lower():
            categorized_type = "resource"
        elif "permission" in str(error).lower() or "unauthorized" in str(error).lower():
            categorized_type = "permission"
        elif "validation" in str(error).lower() or "invalid" in str(error).lower():
            categorized_type = "validation"
        else:
            categorized_type = "unknown"
        
        return {
            "type": categorized_type,
            "original_type": error_type,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now()
        }
    
    async def _record_error(
        self, 
        tool_id: str, 
        error_info: Dict[str, Any], 
        context: ToolcallContext,
        execution_count: int
    ):
        """记录错误"""
        # 更新错误统计
        stats = self.error_statistics[tool_id]
        stats["total_errors"] += 1
        stats["error_types"][error_info["type"]] += 1
        stats["last_error_time"] = datetime.now()
        stats["retry_count"] += execution_count
        
        # 记录错误历史
        self.error_history.append({
            "tool_id": tool_id,
            "error_info": error_info,
            "context": context.to_dict(),
            "execution_count": execution_count,
            "timestamp": time.time()
        })
        
        logger.error(f"记录错误: {tool_id}, 类型: {error_info['type']}")
    
    async def _record_recovery(self, tool_id: str, retry_count: int):
        """记录恢复成功"""
        stats = self.error_statistics[tool_id]
        stats["recovery_count"] += 1
        
        logger.info(f"记录恢复: {tool_id}, 重试次数: {retry_count}")
    
    def _calculate_retry_delay(self, attempt: int, error_info: Dict[str, Any]) -> float:
        """计算重试延迟"""
        base_delay = self.retry_delay
        
        # 指数退避
        exponential_delay = base_delay * (2 ** attempt)
        
        # 根据错误类型调整
        error_type = error_info["type"]
        if error_type == "timeout":
            multiplier = 1.5
        elif error_type == "connection":
            multiplier = 2.0
        else:
            multiplier = 1.0
        
        return min(exponential_delay * multiplier, 60)  # 最大60秒
    
    def _is_circuit_breaker_open(self, tool_id: str) -> bool:
        """检查熔断器是否开启"""
        if tool_id not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[tool_id]
        
        # 检查是否超时
        if datetime.now() - breaker["open_time"] > timedelta(seconds=breaker["timeout"]):
            # 尝试半开状态
            breaker["state"] = "half-open"
            return False
        
        return breaker["state"] == "open"
    
    async def _handle_circuit_breaker(self, tool_id: str, error_info: Dict[str, Any]):
        """处理熔断器"""
        if tool_id not in self.circuit_breakers:
            self.circuit_breakers[tool_id] = {
                "state": "closed",
                "failure_count": 0,
                "failure_threshold": 5,
                "timeout": 60,
                "open_time": None
            }
        
        breaker = self.circuit_breakers[tool_id]
        breaker["failure_count"] += 1
        
        # 检查是否应该开启熔断器
        if breaker["failure_count"] >= breaker["failure_threshold"]:
            breaker["state"] = "open"
            breaker["open_time"] = datetime.now()
            
            logger.warning(f"开启熔断器: {tool_id}")
    
    def _is_error_rate_too_high(self, tool_id: str) -> bool:
        """检查错误率是否过高"""
        # 获取最近的错误记录
        recent_errors = [
            record for record in self.error_history
            if record["tool_id"] == tool_id and 
            time.time() - record["timestamp"] < 300  # 5分钟内
        ]
        
        if len(recent_errors) < 5:  # 样本太少
            return False
        
        # 计算错误率
        error_count = len(recent_errors)
        total_attempts = error_count + 10  # 假设有10次成功尝试
        
        error_rate = error_count / total_attempts
        
        return error_rate > 0.8  # 错误率超过80%
    
    async def _cleanup_resources(self):
        """清理资源"""
        # 清理过期的错误记录
        cutoff_time = time.time() - 3600  # 1小时前
        
        # 清理错误历史
        while self.error_history and self.error_history[0]["timestamp"] < cutoff_time:
            self.error_history.popleft()
        
        # 清理统计信息
        for tool_id in list(self.error_statistics.keys()):
            if self.error_statistics[tool_id]["last_error_time"]:
                if datetime.now() - self.error_statistics[tool_id]["last_error_time"] > timedelta(hours=24):
                    del self.error_statistics[tool_id]
    
    async def _analyze_plan_failures(
        self, 
        plan: ExecutionPlan, 
        failed_tools: List[str], 
        context: ToolcallContext
    ) -> Dict[str, Any]:
        """分析计划失败原因"""
        analysis = {
            "failed_tools": failed_tools,
            "failure_patterns": {},
            "impact_assessment": {},
            "recovery_options": []
        }
        
        # 分析失败模式
        for tool_id in failed_tools:
            tool_stats = self.error_statistics.get(tool_id, {})
            most_common_error = max(
                tool_stats.get("error_types", {}).items(),
                key=lambda x: x[1],
                default=("unknown", 0)
            )
            
            analysis["failure_patterns"][tool_id] = {
                "most_common_error": most_common_error[0],
                "error_count": most_common_error[1],
                "total_errors": tool_stats.get("total_errors", 0)
            }
        
        # 评估影响
        for tool_id in plan.tool_sequence:
            if tool_id in failed_tools:
                # 检查是否有其他工具依赖它
                dependent_tools = [
                    dep_tool for dep_tool, deps in plan.dependencies.items()
                    if tool_id in deps
                ]
                
                analysis["impact_assessment"][tool_id] = {
                    "direct_failure": True,
                    "blocks_tools": dependent_tools,
                    "critical_path": tool_id in plan.tool_sequence[:len(plan.tool_sequence)//2]
                }
        
        return analysis
    
    async def _generate_recovery_plan(
        self, 
        original_plan: ExecutionPlan, 
        failure_analysis: Dict[str, Any]
    ) -> ExecutionPlan:
        """生成恢复计划"""
        recovery_plan = ExecutionPlan(
            tool_sequence=original_plan.tool_sequence.copy(),
            dependencies=original_plan.dependencies.copy(),
            parallel_groups=original_plan.parallel_groups.copy()
        )
        
        failed_tools = failure_analysis["failed_tools"]
        
        # 移除失败的工具（如果它们不是关键路径）
        critical_failed_tools = [
            tool_id for tool_id in failed_tools
            if failure_analysis["impact_assessment"][tool_id].get("critical_path", False)
        ]
        
        if not critical_failed_tools:
            # 非关键路径失败，可以跳过
            recovery_plan.tool_sequence = [
                tool for tool in recovery_plan.tool_sequence
                if tool not in failed_tools
            ]
            
            # 更新依赖关系
            for tool_id, deps in recovery_plan.dependencies.items():
                recovery_plan.dependencies[tool_id] = [
                    dep for dep in deps if dep not in failed_tools
                ]
            
            logger.info(f"生成恢复计划，跳过失败工具: {failed_tools}")
        else:
            # 关键路径失败，需要替代方案
            logger.warning(f"关键路径失败，需要替代方案: {critical_failed_tools}")
        
        return recovery_plan
    
    async def _execute_recovery_plan(
        self, 
        recovery_plan: ExecutionPlan, 
        context: ToolcallContext,
        executor_func: Callable
    ) -> bool:
        """执行恢复计划"""
        try:
            results = []
            for tool_id in recovery_plan.tool_sequence:
                try:
                    result = await executor_func(tool_id, context)
                    results.append(result)
                except Exception as e:
                    logger.error(f"恢复计划中的工具 {tool_id} 也失败: {e}")
                    return False
            
            logger.info("恢复计划执行成功")
            return True
            
        except Exception as e:
            logger.error(f"恢复计划执行失败: {e}")
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        total_errors = sum(stats["total_errors"] for stats in self.error_statistics.values())
        total_recoveries = sum(stats["recovery_count"] for stats in self.error_statistics.values())
        
        # 错误类型分布
        error_type_distribution = defaultdict(int)
        for stats in self.error_statistics.values():
            for error_type, count in stats["error_types"].items():
                error_type_distribution[error_type] += count
        
        # 熔断器状态
        circuit_breaker_status = {}
        for tool_id, breaker in self.circuit_breakers.items():
            circuit_breaker_status[tool_id] = {
                "state": breaker["state"],
                "failure_count": breaker["failure_count"],
                "is_open": breaker["state"] == "open"
            }
        
        return {
            "total_tools_with_errors": len(self.error_statistics),
            "total_errors": total_errors,
            "total_recoveries": total_recoveries,
            "recovery_rate": total_recoveries / max(total_errors, 1),
            "error_type_distribution": dict(error_type_distribution),
            "circuit_breaker_status": circuit_breaker_status,
            "recent_error_rate": self._calculate_recent_error_rate()
        }
    
    def _calculate_recent_error_rate(self) -> float:
        """计算最近的错误率"""
        recent_time = time.time() - 3600  # 1小时内
        
        recent_errors = [
            record for record in self.error_history
            if record["timestamp"] > recent_time
        ]
        
        if not recent_errors:
            return 0.0
        
        error_count = len(recent_errors)
        total_attempts = error_count + 20  # 假设有20次成功尝试
        
        return error_count / total_attempts
    
    async def reset_circuit_breaker(self, tool_id: str):
        """重置熔断器"""
        if tool_id in self.circuit_breakers:
            self.circuit_breakers[tool_id]["state"] = "closed"
            self.circuit_breakers[tool_id]["failure_count"] = 0
            self.circuit_breakers[tool_id]["open_time"] = None
            
            logger.info(f"重置熔断器: {tool_id}")
    
    async def clear_error_history(self):
        """清除错误历史"""
        self.error_history.clear()
        self.error_statistics.clear()
        self.circuit_breakers.clear()
        
        logger.info("清除错误历史")