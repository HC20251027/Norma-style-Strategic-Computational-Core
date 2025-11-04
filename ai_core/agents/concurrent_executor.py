"""
并发工具调用和结果融合系统
支持并行执行多个工具调用并智能融合结果
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import time
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """执行策略枚举"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    CONDITIONAL = "conditional"
    ADAPTIVE = "adaptive"


class ResultStatus(Enum):
    """结果状态枚举"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class ToolCall:
    """工具调用定义"""
    tool_id: str
    tool_name: str
    parameters: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 2
    dependencies: List[str] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """执行结果"""
    tool_id: str
    status: ResultStatus
    result: Any
    execution_time: float
    start_time: datetime
    end_time: datetime
    error: Optional[Exception] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0


@dataclass
class FusionRule:
    """结果融合规则"""
    rule_id: str
    name: str
    description: str
    applicable_types: List[str]
    fusion_strategy: str
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)


class ConcurrentToolExecutor:
    """并发工具执行器"""
    
    def __init__(self, max_workers: int = 10, default_timeout: float = 30.0):
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.tool_registry: Dict[str, Callable] = {}
        self.execution_history: List[ExecutionResult] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.fusion_rules: Dict[str, FusionRule] = {}
        self._lock = threading.Lock()
        
        # 设置默认融合规则
        self._setup_default_fusion_rules()
    
    def _setup_default_fusion_rules(self):
        """设置默认融合规则"""
        default_rules = [
            FusionRule(
                rule_id="merge_dicts",
                name="字典合并",
                description="合并多个字典结果",
                applicable_types=["dict"],
                fusion_strategy="merge",
                priority=1
            ),
            FusionRule(
                rule_id="combine_lists",
                name="列表合并",
                description="合并多个列表结果",
                applicable_types=["list"],
                fusion_strategy="concatenate",
                priority=1
            ),
            FusionRule(
                rule_id="weighted_average",
                name="加权平均",
                description="对数值结果进行加权平均",
                applicable_types=["number"],
                fusion_strategy="weighted_avg",
                priority=2
            ),
            FusionRule(
                rule_id="consensus",
                name="共识融合",
                description="基于置信度的共识融合",
                applicable_types=["any"],
                fusion_strategy="consensus",
                priority=3
            )
        ]
        
        for rule in default_rules:
            self.fusion_rules[rule.rule_id] = rule
    
    def register_tool(self, tool_name: str, tool_function: Callable):
        """注册工具函数"""
        self.tool_registry[tool_name] = tool_function
        logger.info(f"注册工具: {tool_name}")
    
    async def execute_single_tool(self, tool_call: ToolCall) -> ExecutionResult:
        """执行单个工具调用"""
        start_time = datetime.now()
        
        try:
            if tool_call.tool_name not in self.tool_registry:
                raise ValueError(f"工具未注册: {tool_call.tool_name}")
            
            tool_function = self.tool_registry[tool_call.tool_name]
            
            # 设置超时
            timeout = tool_call.timeout or self.default_timeout
            
            # 执行工具调用
            if asyncio.iscoroutinefunction(tool_function):
                result = await asyncio.wait_for(tool_function(**tool_call.parameters), timeout=timeout)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: tool_function(**tool_call.parameters)),
                    timeout=timeout
                )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 计算置信度分数
            confidence_score = self._calculate_confidence_score(result, tool_call)
            
            execution_result = ExecutionResult(
                tool_id=tool_call.tool_id,
                status=ResultStatus.SUCCESS,
                result=result,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                confidence_score=confidence_score,
                metadata=tool_call.metadata
            )
            
            # 更新性能指标
            self._update_performance_metrics(tool_call.tool_name, execution_time, True)
            
            return execution_result
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return ExecutionResult(
                tool_id=tool_call.tool_id,
                status=ResultStatus.TIMEOUT,
                result=None,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error=TimeoutError(f"工具执行超时: {tool_call.timeout}秒"),
                retry_count=tool_call.retry_count
            )
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # 检查是否需要重试
            if tool_call.retry_count < tool_call.max_retries:
                tool_call.retry_count += 1
                logger.warning(f"工具执行失败，准备重试: {tool_call.tool_name}, 错误: {str(e)}")
                return await self.execute_single_tool(tool_call)
            
            self._update_performance_metrics(tool_call.tool_name, execution_time, False)
            
            return ExecutionResult(
                tool_id=tool_call.tool_id,
                status=ResultStatus.FAILURE,
                result=None,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                error=e,
                retry_count=tool_call.retry_count
            )
    
    def _calculate_confidence_score(self, result: Any, tool_call: ToolCall) -> float:
        """计算结果置信度分数"""
        base_score = 0.5
        
        # 基于结果类型调整分数
        if result is None:
            return 0.0
        
        if isinstance(result, dict):
            # 字典结果，根据键值对数量调整
            if len(result) > 0:
                base_score = min(0.9, 0.5 + len(result) * 0.1)
            else:
                base_score = 0.3
        
        elif isinstance(result, list):
            # 列表结果，根据长度调整
            if len(result) > 0:
                base_score = min(0.9, 0.5 + len(result) * 0.05)
            else:
                base_score = 0.3
        
        elif isinstance(result, (int, float)):
            # 数值结果，假设在合理范围内
            base_score = 0.8
        
        elif isinstance(result, str):
            # 字符串结果，根据长度调整
            if len(result) > 10:
                base_score = 0.7
            else:
                base_score = 0.5
        
        # 基于工具可靠性调整
        if tool_call.tool_name in self.performance_metrics:
            avg_success_rate = self.performance_metrics[tool_call.tool_name].get('success_rate', 0.9)
            base_score *= avg_success_rate
        
        return min(base_score, 1.0)
    
    def _update_performance_metrics(self, tool_name: str, execution_time: float, success: bool):
        """更新性能指标"""
        with self._lock:
            if tool_name not in self.performance_metrics:
                self.performance_metrics[tool_name] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'success_rate': 0.0
                }
            
            metrics = self.performance_metrics[tool_name]
            metrics['total_executions'] += 1
            
            if success:
                metrics['successful_executions'] += 1
            
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_executions']
            metrics['success_rate'] = metrics['successful_executions'] / metrics['total_executions']
    
    async def execute_tools_sequential(self, tool_calls: List[ToolCall]) -> List[ExecutionResult]:
        """顺序执行工具调用"""
        results = []
        
        for tool_call in tool_calls:
            result = await self.execute_single_tool(tool_call)
            results.append(result)
            
            # 如果执行失败，可以选择停止或继续
            if result.status == ResultStatus.FAILURE and tool_call.metadata.get('stop_on_failure', False):
                logger.warning(f"工具执行失败，停止执行: {tool_call.tool_name}")
                break
        
        return results
    
    async def execute_tools_parallel(self, tool_calls: List[ToolCall]) -> List[ExecutionResult]:
        """并行执行工具调用"""
        # 创建任务
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(self.execute_single_tool(tool_call))
            self.active_executions[tool_call.tool_id] = task
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 清理活跃执行记录
        for tool_call in tool_calls:
            self.active_executions.pop(tool_call.tool_id, None)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ExecutionResult(
                    tool_id=tool_calls[i].tool_id,
                    status=ResultStatus.FAILURE,
                    result=None,
                    execution_time=0.0,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error=result
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_tools_pipeline(self, tool_calls: List[ToolCall]) -> List[ExecutionResult]:
        """流水线执行工具调用（带依赖关系）"""
        results = []
        completed_tools = set()
        
        # 按依赖关系排序
        sorted_calls = self._topological_sort(tool_calls)
        
        for tool_call in sorted_calls:
            # 检查依赖是否完成
            if not all(dep in completed_tools for dep in tool_call.dependencies):
                # 等待依赖完成
                await asyncio.sleep(0.1)
                if not all(dep in completed_tools for dep in tool_call.dependencies):
                    logger.warning(f"依赖未满足，跳过工具: {tool_call.tool_name}")
                    continue
            
            result = await self.execute_single_tool(tool_call)
            results.append(result)
            
            if result.status == ResultStatus.SUCCESS:
                completed_tools.add(tool_call.tool_id)
                
                # 将结果传递给依赖此工具的其他工具
                for other_call in tool_calls:
                    if tool_call.tool_id in other_call.dependencies:
                        other_call.execution_context['dependency_results'] = other_call.execution_context.get('dependency_results', {})
                        other_call.execution_context['dependency_results'][tool_call.tool_id] = result.result
        
        return results
    
    def _topological_sort(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        """拓扑排序"""
        # 构建依赖图
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for tool_call in tool_calls:
            in_degree[tool_call.tool_id] = 0
        
        for tool_call in tool_calls:
            for dep in tool_call.dependencies:
                graph[dep].append(tool_call.tool_id)
                in_degree[tool_call.tool_id] += 1
        
        # 拓扑排序
        queue = [tool_id for tool_id in in_degree if in_degree[tool_id] == 0]
        sorted_calls = []
        
        while queue:
            current = queue.pop(0)
            # 找到对应的tool_call
            current_call = next((call for call in tool_calls if call.tool_id == current), None)
            if current_call:
                sorted_calls.append(current_call)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return sorted_calls
    
    async def execute_tools_adaptive(self, tool_calls: List[ToolCall]) -> List[ExecutionResult]:
        """自适应执行策略"""
        # 分析工具调用特征
        analysis = self._analyze_tool_calls(tool_calls)
        
        # 根据分析结果选择执行策略
        if analysis['can_parallelize'] and len(tool_calls) > 2:
            logger.info("选择并行执行策略")
            return await self.execute_tools_parallel(tool_calls)
        elif analysis['has_dependencies']:
            logger.info("选择流水线执行策略")
            return await self.execute_tools_pipeline(tool_calls)
        else:
            logger.info("选择顺序执行策略")
            return await self.execute_tools_sequential(tool_calls)
    
    def _analyze_tool_calls(self, tool_calls: List[ToolCall]) -> Dict[str, Any]:
        """分析工具调用特征"""
        analysis = {
            'can_parallelize': True,
            'has_dependencies': False,
            'estimated_total_time': 0.0,
            'resource_intensive': False
        }
        
        for tool_call in tool_calls:
            # 检查依赖关系
            if tool_call.dependencies:
                analysis['has_dependencies'] = True
                analysis['can_parallelize'] = False
            
            # 估算执行时间
            tool_name = tool_call.tool_name
            if tool_name in self.performance_metrics:
                avg_time = self.performance_metrics[tool_name].get('avg_time', 1.0)
                analysis['estimated_total_time'] += avg_time
                
                # 检查是否为资源密集型
                if avg_time > 5.0:
                    analysis['resource_intensive'] = True
        
        return analysis
    
    def fuse_results(self, results: List[ExecutionResult], fusion_strategy: str = "auto") -> Any:
        """融合执行结果"""
        # 过滤成功的结果
        successful_results = [r for r in results if r.status == ResultStatus.SUCCESS]
        
        if not successful_results:
            return None
        
        if len(successful_results) == 1:
            return successful_results[0].result
        
        # 自动选择融合策略
        if fusion_strategy == "auto":
            fusion_strategy = self._select_fusion_strategy(successful_results)
        
        # 执行融合
        return self._apply_fusion_strategy(successful_results, fusion_strategy)
    
    def _select_fusion_strategy(self, results: List[ExecutionResult]) -> str:
        """自动选择融合策略"""
        result_types = set(type(r.result).__name__ for r in results)
        
        if len(result_types) == 1:
            result_type = list(result_types)[0]
            
            # 根据结果类型选择策略
            if result_type == 'dict':
                return 'merge'
            elif result_type == 'list':
                return 'concatenate'
            elif result_type == 'int' or result_type == 'float':
                return 'weighted_avg'
            else:
                return 'consensus'
        
        return 'consensus'
    
    def _apply_fusion_strategy(self, results: List[ExecutionResult], strategy: str) -> Any:
        """应用融合策略"""
        if strategy == "merge":
            return self._merge_dicts(results)
        elif strategy == "concatenate":
            return self._concatenate_lists(results)
        elif strategy == "weighted_avg":
            return self._weighted_average(results)
        elif strategy == "consensus":
            return self._consensus_fusion(results)
        else:
            return results[0].result  # 默认返回第一个结果
    
    def _merge_dicts(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """合并字典结果"""
        merged = {}
        for result in results:
            if isinstance(result.result, dict):
                merged.update(result.result)
        return merged
    
    def _concatenate_lists(self, results: List[ExecutionResult]) -> List[Any]:
        """合并列表结果"""
        concatenated = []
        for result in results:
            if isinstance(result.result, list):
                concatenated.extend(result.result)
        return concatenated
    
    def _weighted_average(self, results: List[ExecutionResult]) -> float:
        """计算加权平均"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for result in results:
            if isinstance(result.result, (int, float)):
                weight = result.confidence_score
                total_weight += weight
                weighted_sum += result.result * weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _consensus_fusion(self, results: List[ExecutionResult]) -> Any:
        """共识融合"""
        # 按置信度排序
        sorted_results = sorted(results, key=lambda x: x.confidence_score, reverse=True)
        
        # 选择置信度最高的结果
        best_result = sorted_results[0]
        
        # 如果有多个高置信度的相似结果，进行合并
        high_confidence_threshold = 0.7
        high_confidence_results = [r for r in sorted_results if r.confidence_score > high_confidence_threshold]
        
        if len(high_confidence_results) > 1:
            # 尝试合并相似结果
            return self._merge_similar_results(high_confidence_results)
        
        return best_result.result
    
    def _merge_similar_results(self, results: List[ExecutionResult]) -> Any:
        """合并相似结果"""
        if not results:
            return None
        
        # 简单的相似性检测和合并
        first_result = results[0].result
        
        if isinstance(first_result, dict):
            return self._merge_dicts(results)
        elif isinstance(first_result, list):
            return self._concatenate_lists(results)
        else:
            # 对于其他类型，返回最高置信度的结果
            return max(results, key=lambda x: x.confidence_score).result
    
    def cancel_execution(self, tool_id: str) -> bool:
        """取消执行"""
        if tool_id in self.active_executions:
            task = self.active_executions[tool_id]
            task.cancel()
            del self.active_executions[tool_id]
            logger.info(f"取消工具执行: {tool_id}")
            return True
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return dict(self.performance_metrics)
    
    def get_execution_history(self, limit: int = 100) -> List[ExecutionResult]:
        """获取执行历史"""
        return self.execution_history[-limit:]
    
    def export_metrics(self) -> Dict[str, Any]:
        """导出指标数据"""
        return {
            'performance_metrics': dict(self.performance_metrics),
            'execution_count': len(self.execution_history),
            'active_executions': len(self.active_executions),
            'registered_tools': list(self.tool_registry.keys()),
            'fusion_rules': {rule_id: {
                'name': rule.name,
                'strategy': rule.fusion_strategy,
                'priority': rule.priority
            } for rule_id, rule in self.fusion_rules.items()}
        }