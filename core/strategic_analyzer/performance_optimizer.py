"""
性能优化器

负责工具调用的性能优化，包括缓存、并行执行、自适应超时等
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import hashlib

from ..models import OptimizationConfig, ExecutionPlan, ToolcallContext


logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # 缓存相关
        self.result_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_access_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 性能指标
        self.performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.execution_history: deque = deque(maxlen=1000)
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(self.config.max_parallel_tools)
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # 自适应超时
        self.timeout_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
    async def optimize_execution(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext,
        executor_func: Callable
    ) -> List[Dict[str, Any]]:
        """优化执行计划"""
        start_time = time.time()
        
        try:
            # 1. 预优化分析
            optimized_plan = await self._pre_optimize_plan(plan, context)
            
            # 2. 并行执行优化
            if self.config.enable_parallel_execution and optimized_plan.parallel_groups:
                results = await self._execute_parallel_groups(optimized_plan, context, executor_func)
            else:
                results = await self._execute_sequential(optimized_plan, context, executor_func)
            
            # 3. 后优化处理
            final_results = await self._post_optimize_results(results, context)
            
            # 4. 更新性能指标
            execution_time = time.time() - start_time
            await self._update_performance_metrics(optimized_plan, execution_time, final_results)
            
            logger.info(f"执行优化完成，耗时: {execution_time:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"执行优化失败: {e}")
            # 回退到原始计划执行
            return await self._execute_fallback(plan, context, executor_func)
    
    async def _pre_optimize_plan(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> ExecutionPlan:
        """预优化执行计划"""
        optimized_plan = plan
        
        # 1. 缓存命中优化
        if self.config.enable_caching:
            optimized_plan = await self._optimize_with_caching(optimized_plan, context)
        
        # 2. 并行分组优化
        if self.config.enable_parallel_execution:
            optimized_plan = await self._optimize_parallel_groups(optimized_plan, context)
        
        # 3. 资源预分配优化
        optimized_plan = await self._optimize_resource_allocation(optimized_plan, context)
        
        # 4. 执行顺序优化
        optimized_plan = await self._optimize_execution_order(optimized_plan, context)
        
        return optimized_plan
    
    async def _optimize_with_caching(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> ExecutionPlan:
        """缓存优化"""
        cached_tools = []
        uncached_tools = []
        
        for tool_id in plan.tool_sequence:
            cache_key = self._generate_cache_key(tool_id, context)
            
            if self._is_cache_valid(cache_key):
                cached_tools.append(tool_id)
                logger.debug(f"工具 {tool_id} 缓存命中")
            else:
                uncached_tools.append(tool_id)
        
        # 如果大部分工具都有缓存，可以考虑跳过执行
        if len(cached_tools) / len(plan.tool_sequence) > self.config.prefetch_threshold:
            logger.info(f"缓存命中率高({len(cached_tools)}/{len(plan.tool_sequence)})，考虑优化执行策略")
        
        return plan
    
    async def _optimize_parallel_groups(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> ExecutionPlan:
        """并行分组优化"""
        if not plan.parallel_groups:
            # 自动生成并行分组
            plan.parallel_groups = self._generate_parallel_groups(plan.tool_sequence, plan.dependencies)
        
        # 优化分组大小
        optimized_groups = []
        current_group = []
        
        for group in plan.parallel_groups:
            if len(current_group) + len(group) <= self.config.max_parallel_tools:
                current_group.extend(group)
            else:
                if current_group:
                    optimized_groups.append(current_group)
                current_group = group.copy()
        
        if current_group:
            optimized_groups.append(current_group)
        
        plan.parallel_groups = optimized_groups
        
        logger.debug(f"并行分组优化: {len(optimized_groups)}个组")
        return plan
    
    async def _optimize_resource_allocation(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> ExecutionPlan:
        """资源分配优化"""
        # 预估资源需求
        resource_requirements = {
            "memory": self._estimate_memory_usage(plan),
            "cpu": self._estimate_cpu_usage(plan),
            "concurrent_tools": len(plan.tool_sequence)
        }
        
        plan.resource_requirements.update(resource_requirements)
        
        return plan
    
    async def _optimize_execution_order(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext
    ) -> ExecutionPlan:
        """执行顺序优化"""
        # 基于历史性能数据调整执行顺序
        tool_performance = self._get_tool_performance_metrics()
        
        # 按性能排序（高性能工具优先）
        sorted_tools = sorted(
            plan.tool_sequence,
            key=lambda x: tool_performance.get(x, {}).get("success_rate", 0.5),
            reverse=True
        )
        
        plan.tool_sequence = sorted_tools
        
        return plan
    
    async def _execute_parallel_groups(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext,
        executor_func: Callable
    ) -> List[Dict[str, Any]]:
        """执行并行分组"""
        all_results = []
        
        for group in plan.parallel_groups:
            # 创建组内并发任务
            tasks = []
            
            for tool_id in group:
                task = asyncio.create_task(
                    self._execute_with_optimization(tool_id, context, executor_func)
                )
                tasks.append(task)
            
            # 等待组内所有任务完成
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            for i, result in enumerate(group_results):
                if isinstance(result, Exception):
                    logger.error(f"组内工具 {group[i]} 执行失败: {result}")
                    all_results.append({
                        "tool_id": group[i],
                        "success": False,
                        "error": str(result)
                    })
                else:
                    all_results.append(result)
        
        return all_results
    
    async def _execute_sequential(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext,
        executor_func: Callable
    ) -> List[Dict[str, Any]]:
        """顺序执行"""
        results = []
        
        for tool_id in plan.tool_sequence:
            result = await self._execute_with_optimization(tool_id, context, executor_func)
            results.append(result)
        
        return results
    
    async def _execute_with_optimization(
        self, 
        tool_id: str, 
        context: ToolcallContext,
        executor_func: Callable
    ) -> Dict[str, Any]:
        """带优化的工具执行"""
        # 获取自适应超时
        timeout = await self._get_adaptive_timeout(tool_id)
        
        # 检查缓存
        cache_key = self._generate_cache_key(tool_id, context)
        cached_result = await self._get_cached_result(cache_key)
        
        if cached_result is not None:
            logger.info(f"工具 {tool_id} 使用缓存结果")
            return cached_result
        
        # 执行工具
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    executor_func(tool_id, context),
                    timeout=timeout
                )
                
                # 缓存结果
                if self.config.enable_caching:
                    await self._cache_result(cache_key, result)
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"工具 {tool_id} 执行超时: {timeout}s")
                return {
                    "tool_id": tool_id,
                    "success": False,
                    "error": f"执行超时: {timeout}s"
                }
            except Exception as e:
                logger.error(f"工具 {tool_id} 执行失败: {e}")
                return {
                    "tool_id": tool_id,
                    "success": False,
                    "error": str(e)
                }
    
    async def _post_optimize_results(
        self, 
        results: List[Dict[str, Any]], 
        context: ToolcallContext
    ) -> List[Dict[str, Any]]:
        """后优化处理"""
        optimized_results = []
        
        for result in results:
            # 结果压缩
            if result.get("success") and isinstance(result.get("data"), dict):
                result["data"] = self._compress_result_data(result["data"])
            
            # 结果验证
            if not self._validate_result(result):
                logger.warning(f"结果验证失败: {result.get('tool_id')}")
                result["success"] = False
                result["error"] = "结果验证失败"
            
            optimized_results.append(result)
        
        return optimized_results
    
    async def _execute_fallback(
        self, 
        plan: ExecutionPlan, 
        context: ToolcallContext,
        executor_func: Callable
    ) -> List[Dict[str, Any]]:
        """回退执行"""
        logger.info("执行回退策略")
        
        results = []
        for tool_id in plan.tool_sequence:
            try:
                result = await executor_func(tool_id, context)
                results.append(result)
            except Exception as e:
                results.append({
                    "tool_id": tool_id,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def _get_adaptive_timeout(self, tool_id: str) -> float:
        """获取自适应超时时间"""
        if not self.config.enable_adaptive_timeout:
            return self.config.base_timeout
        
        # 基于历史数据调整超时
        history = self.timeout_history[tool_id]
        
        if not history:
            return self.config.base_timeout
        
        # 计算平均执行时间
        avg_time = sum(history) / len(history)
        
        # 自适应调整
        adaptive_timeout = min(
            avg_time * self.config.timeout_multiplier,
            self.config.base_timeout * 3  # 最大不超过基础超时的3倍
        )
        
        return max(adaptive_timeout, 5)  # 最小5秒
    
    def _generate_cache_key(self, tool_id: str, context: ToolcallContext) -> str:
        """生成缓存键"""
        content = f"{tool_id}:{context.request_id}:{str(sorted(context.metadata.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.result_cache:
            return False
        
        cache_entry = self.result_cache[cache_key]
        cache_time = cache_entry.get("timestamp")
        
        if not cache_time:
            return False
        
        # 检查是否过期
        return datetime.now() - cache_time < timedelta(seconds=self.config.cache_ttl)
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        if cache_key in self.result_cache:
            # 更新访问时间
            self.cache_access_times[cache_key].append(time.time())
            return self.result_cache[cache_key]["result"]
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """缓存结果"""
        self.result_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now(),
            "access_count": 1
        }
        
        # 清理过期缓存
        await self._cleanup_expired_cache()
    
    async def _cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, cache_entry in self.result_cache.items():
            if current_time - cache_entry["timestamp"] > timedelta(seconds=self.config.cache_ttl):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.result_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
        
        if expired_keys:
            logger.debug(f"清理过期缓存: {len(expired_keys)}条")
    
    def _generate_parallel_groups(
        self, 
        tool_sequence: List[str], 
        dependencies: Dict[str, List[str]]
    ) -> List[List[str]]:
        """生成并行分组"""
        # 简单的拓扑排序和分组算法
        groups = []
        remaining_tools = set(tool_sequence)
        processed_tools = set()
        
        while remaining_tools:
            # 找到没有未处理依赖的工具
            ready_tools = []
            for tool in remaining_tools:
                deps = dependencies.get(tool, [])
                if all(dep in processed_tools for dep in deps):
                    ready_tools.append(tool)
            
            if not ready_tools:
                # 如果没有就绪工具，选择第一个（避免死锁）
                ready_tools = [list(remaining_tools)[0]]
            
            groups.append(ready_tools)
            
            # 更新集合
            remaining_tools -= set(ready_tools)
            processed_tools.update(ready_tools)
        
        return groups
    
    def _estimate_memory_usage(self, plan: ExecutionPlan) -> str:
        """预估内存使用"""
        # 简化的内存估算
        base_memory = 100  # MB
        per_tool_memory = 50  # MB
        
        estimated = base_memory + len(plan.tool_sequence) * per_tool_memory
        
        if estimated < 1024:
            return f"{estimated}MB"
        else:
            return f"{estimated/1024:.1f}GB"
    
    def _estimate_cpu_usage(self, plan: ExecutionPlan) -> str:
        """预估CPU使用"""
        # 简化的CPU估算
        base_cpu = 10  # %
        per_tool_cpu = 20  # %
        
        estimated = min(base_cpu + len(plan.tool_sequence) * per_tool_cpu, 100)
        
        return f"{estimated}%"
    
    def _compress_result_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """压缩结果数据"""
        # 简单的数据压缩逻辑
        compressed = {}
        
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 1000:
                # 长字符串截断
                compressed[key] = value[:997] + "..."
            elif isinstance(value, list) and len(value) > 100:
                # 大列表截断
                compressed[key] = value[:97] + ["..."]
            elif isinstance(value, dict) and len(value) > 50:
                # 大字典截断
                items = list(value.items())[:47]
                compressed[key] = dict(items + [("...", "...")])
            else:
                compressed[key] = value
        
        return compressed
    
    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """验证结果"""
        if not isinstance(result, dict):
            return False
        
        if "tool_id" not in result:
            return False
        
        if "success" not in result:
            return False
        
        if result["success"] and "data" not in result:
            return False
        
        if not result["success"] and "error" not in result:
            return False
        
        return True
    
    def _get_tool_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """获取工具性能指标"""
        metrics = {}
        
        for record in self.execution_history:
            tool_id = record.get("tool_id")
            if not tool_id:
                continue
            
            if tool_id not in metrics:
                metrics[tool_id] = {
                    "success_count": 0,
                    "total_count": 0,
                    "total_time": 0.0
                }
            
            tool_metrics = metrics[tool_id]
            tool_metrics["total_count"] += 1
            
            if record.get("success", False):
                tool_metrics["success_count"] += 1
            
            tool_metrics["total_time"] += record.get("execution_time", 0.0)
        
        # 计算成功率
        for tool_id in metrics:
            tool_metrics = metrics[tool_id]
            tool_metrics["success_rate"] = (
                tool_metrics["success_count"] / 
                max(tool_metrics["total_count"], 1)
            )
            tool_metrics["average_time"] = (
                tool_metrics["total_time"] / 
                max(tool_metrics["total_count"], 1)
            )
        
        return metrics
    
    async def _update_performance_metrics(
        self, 
        plan: ExecutionPlan, 
        execution_time: float, 
        results: List[Dict[str, Any]]
    ):
        """更新性能指标"""
        # 记录执行历史
        for result in results:
            self.execution_history.append({
                "tool_id": result.get("tool_id"),
                "success": result.get("success", False),
                "execution_time": result.get("execution_time", 0.0),
                "timestamp": time.time()
            })
            
            # 更新超时历史
            if result.get("success"):
                exec_time = result.get("execution_time", 0.0)
                if exec_time > 0:
                    tool_id = result.get("tool_id")
                    if tool_id:
                        self.timeout_history[tool_id].append(exec_time)
        
        # 清理历史记录
        while len(self.execution_history) > 1000:
            self.execution_history.popleft()
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        metrics = self._get_tool_performance_metrics()
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for record in self.execution_history if record.get("success", False))
        
        cache_hit_rate = 0.0
        if self.cache_access_times:
            total_cache_accesses = sum(len(access_times) for access_times in self.cache_access_times.values())
            cache_hit_rate = total_cache_accesses / max(total_executions, 1)
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / max(total_executions, 1),
            "cache_hit_rate": cache_hit_rate,
            "tool_performance": metrics,
            "cache_size": len(self.result_cache),
            "active_executions": len(self.active_executions)
        }
    
    async def optimize_configuration(self) -> OptimizationConfig:
        """基于历史数据优化配置"""
        # 分析性能数据，优化配置参数
        metrics = self.get_performance_statistics()
        
        # 如果成功率较低，增加重试次数
        if metrics["success_rate"] < 0.8:
            self.config.max_retries = min(self.config.max_retries + 1, 5)
        
        # 如果缓存命中率低，延长缓存时间
        if metrics["cache_hit_rate"] < 0.3:
            self.config.cache_ttl = min(self.config.cache_ttl * 1.5, 7200)  # 最大2小时
        
        # 如果平均执行时间较长，增加超时时间
        avg_execution_time = sum(
            record.get("execution_time", 0) 
            for record in self.execution_history
        ) / max(len(self.execution_history), 1)
        
        if avg_execution_time > 10:  # 超过10秒
            self.config.base_timeout = min(self.config.base_timeout * 1.2, 120)
        
        logger.info("配置优化完成")
        return self.config