"""
任务结果合并和融合
负责收集、合并、聚合多个任务的执行结果
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Generic, TypeVar, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from abc import ABC, abstractmethod

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')


class MergeStrategy(Enum):
    """合并策略"""
    FIRST = "first"
    LAST = "last"
    CONCAT = "concat"
    UNION = "union"
    INTERSECTION = "intersection"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    CUSTOM = "custom"


class AggregationType(Enum):
    """聚合类型"""
    LIST = "list"
    DICT = "dict"
    SET = "set"
    COUNTER = "counter"
    STATISTICS = "statistics"
    CUSTOM = "custom"


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    result: Any
    status: str  # success, failed, timeout
    execution_time: float
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    error: Optional[Exception] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class MergeRule:
    """合并规则"""
    strategy: MergeStrategy
    aggregation_type: AggregationType = AggregationType.LIST
    custom_function: Optional[Callable] = None
    priority: int = 1  # 优先级，数字越大优先级越高
    condition: Optional[Callable] = None  # 条件函数
    metadata: dict = field(default_factory=dict)


class ResultMerger:
    """结果合并器"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 结果存储
        self.results: Dict[str, TaskResult] = {}
        self.merge_rules: Dict[str, MergeRule] = {}
        self.merged_results: Dict[str, Any] = {}
        
        # 合并统计
        self.stats = {
            'total_merges': 0,
            'successful_merges': 0,
            'failed_merges': 0,
            'avg_merge_time': 0.0,
            'total_merge_time': 0.0
        }
        
        self._lock = threading.RLock()
        
        logger.info("结果合并器初始化完成")
    
    def add_result(self, result: TaskResult):
        """添加任务结果"""
        with self._lock:
            self.results[result.task_id] = result
            logger.debug(f"任务结果已添加: {result.task_id}")
    
    def add_results_batch(self, results: List[TaskResult]):
        """批量添加任务结果"""
        with self._lock:
            for result in results:
                self.results[result.task_id] = result
            
            logger.info(f"批量添加任务结果完成，共 {len(results)} 个结果")
    
    def register_merge_rule(self, rule_name: str, rule: MergeRule):
        """注册合并规则"""
        with self._lock:
            self.merge_rules[rule_name] = rule
            logger.info(f"合并规则已注册: {rule_name}")
    
    async def merge_results(
        self,
        task_ids: List[str],
        rule_name: str = "default",
        custom_aggregation: Optional[Callable] = None
    ) -> Any:
        """合并结果"""
        start_time = time.time()
        self.stats['total_merges'] += 1
        
        try:
            # 获取结果
            results = []
            for task_id in task_ids:
                if task_id in self.results:
                    results.append(self.results[task_id])
                else:
                    logger.warning(f"任务结果不存在: {task_id}")
            
            if not results:
                raise ValueError("没有可合并的结果")
            
            # 检查合并规则
            if rule_name not in self.merge_rules:
                raise ValueError(f"合并规则不存在: {rule_name}")
            
            rule = self.merge_rules[rule_name]
            
            # 检查条件
            if rule.condition and not rule.condition(results):
                logger.info(f"合并条件不满足，跳过合并")
                return None
            
            # 执行合并
            merged_result = await self._execute_merge(results, rule, custom_aggregation)
            
            # 记录统计
            merge_time = time.time() - start_time
            self.stats['successful_merges'] += 1
            self.stats['total_merge_time'] += merge_time
            self.stats['avg_merge_time'] = (
                self.stats['total_merge_time'] / self.stats['successful_merges']
            )
            
            # 缓存结果
            cache_key = f"{rule_name}_{hash(tuple(sorted(task_ids)))}"
            self.merged_results[cache_key] = merged_result
            
            logger.info(f"结果合并完成，使用规则: {rule_name}, 耗时: {merge_time:.2f}s")
            return merged_result
            
        except Exception as e:
            self.stats['failed_merges'] += 1
            logger.error(f"结果合并失败: {e}")
            raise
    
    async def _execute_merge(
        self,
        results: List[TaskResult],
        rule: MergeRule,
        custom_aggregation: Optional[Callable] = None
    ) -> Any:
        """执行合并逻辑"""
        # 过滤成功的结果
        successful_results = [r for r in results if r.status == "success"]
        
        if not successful_results:
            logger.warning("没有成功的任务结果可合并")
            return None
        
        # 根据聚合类型处理
        if rule.aggregation_type == AggregationType.LIST:
            return await self._merge_as_list(successful_results, rule, custom_aggregation)
        elif rule.aggregation_type == AggregationType.DICT:
            return await self._merge_as_dict(successful_results, rule, custom_aggregation)
        elif rule.aggregation_type == AggregationType.SET:
            return await self._merge_as_set(successful_results, rule, custom_aggregation)
        elif rule.aggregation_type == AggregationType.COUNTER:
            return await self._merge_as_counter(successful_results, rule, custom_aggregation)
        elif rule.aggregation_type == AggregationType.STATISTICS:
            return await self._merge_as_statistics(successful_results, rule, custom_aggregation)
        else:
            return await self._merge_custom(successful_results, rule, custom_aggregation)
    
    async def _merge_as_list(
        self,
        results: List[TaskResult],
        rule: MergeRule,
        custom_aggregation: Optional[Callable] = None
    ) -> List[Any]:
        """作为列表合并"""
        values = [r.result for r in results]
        
        if custom_aggregation:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, custom_aggregation, values
            )
        
        if rule.strategy == MergeStrategy.FIRST:
            return [values[0]]
        elif rule.strategy == MergeStrategy.LAST:
            return [values[-1]]
        elif rule.strategy == MergeStrategy.CONCAT:
            # 展平列表
            flattened = []
            for value in values:
                if isinstance(value, list):
                    flattened.extend(value)
                else:
                    flattened.append(value)
            return flattened
        elif rule.strategy == MergeStrategy.UNION:
            # 获取所有唯一值
            all_values = []
            for value in values:
                if isinstance(value, (list, set)):
                    all_values.extend(value)
                else:
                    all_values.append(value)
            return list(set(all_values))
        else:
            # 默认返回所有结果
            return values
    
    async def _merge_as_dict(
        self,
        results: List[TaskResult],
        rule: MergeRule,
        custom_aggregation: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """作为字典合并"""
        dicts = [r.result for r in results if isinstance(r.result, dict)]
        
        if not dicts:
            return {}
        
        if custom_aggregation:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, custom_aggregation, dicts
            )
        
        merged = {}
        
        if rule.strategy == MergeStrategy.UNION:
            # 合并所有键值对
            for d in dicts:
                merged.update(d)
        elif rule.strategy == MergeStrategy.INTERSECTION:
            # 获取共同键
            if dicts:
                common_keys = set(dicts[0].keys())
                for d in dicts[1:]:
                    common_keys &= set(d.keys())
                
                for key in common_keys:
                    merged[key] = [d[key] for d in dicts if key in d]
        else:
            # 默认策略：使用最后一个字典
            merged = dicts[-1]
        
        return merged
    
    async def _merge_as_set(
        self,
        results: List[TaskResult],
        rule: MergeRule,
        custom_aggregation: Optional[Callable] = None
    ) -> Set[Any]:
        """作为集合合并"""
        sets = []
        for r in results:
            if isinstance(r.result, set):
                sets.append(r.result)
            elif isinstance(r.result, (list, tuple)):
                sets.append(set(r.result))
            else:
                sets.append({r.result})
        
        if custom_aggregation:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, custom_aggregation, sets
            )
        
        if rule.strategy == MergeStrategy.UNION:
            result = set()
            for s in sets:
                result |= s
            return result
        elif rule.strategy == MergeStrategy.INTERSECTION:
            if sets:
                result = sets[0].copy()
                for s in sets[1:]:
                    result &= s
                return result
        else:
            # 默认返回第一个集合
            return sets[0] if sets else set()
        
        return set()
    
    async def _merge_as_counter(
        self,
        results: List[TaskResult],
        rule: MergeRule,
        custom_aggregation: Optional[Callable] = None
    ) -> Counter:
        """作为计数器合并"""
        counters = []
        for r in results:
            if isinstance(r.result, Counter):
                counters.append(r.result)
            elif isinstance(r.result, dict):
                counters.append(Counter(r.result))
            else:
                counters.append(Counter([r.result]))
        
        if custom_aggregation:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, custom_aggregation, counters
            )
        
        # 默认合并：求和
        merged = Counter()
        for counter in counters:
            merged += counter
        
        return merged
    
    async def _merge_as_statistics(
        self,
        results: List[TaskResult],
        rule: MergeRule,
        custom_aggregation: Optional[Callable] = None
    ) -> Dict[str, float]:
        """作为统计信息合并"""
        numeric_values = []
        for r in results:
            if isinstance(r.result, (int, float)):
                numeric_values.append(r.result)
            elif isinstance(r.result, (list, tuple)):
                numeric_values.extend([v for v in r.result if isinstance(v, (int, float))])
        
        if not numeric_values:
            return {}
        
        if custom_aggregation:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, custom_aggregation, numeric_values
            )
        
        # 计算统计信息
        stats = {}
        
        if rule.strategy == MergeStrategy.SUM:
            stats['sum'] = sum(numeric_values)
        elif rule.strategy == MergeStrategy.AVERAGE:
            stats['average'] = np.mean(numeric_values)
        elif rule.strategy == MergeStrategy.MIN:
            stats['min'] = min(numeric_values)
        elif rule.strategy == MergeStrategy.MAX:
            stats['max'] = max(numeric_values)
        elif rule.strategy == MergeStrategy.MEDIAN:
            stats['median'] = np.median(numeric_values)
        else:
            # 完整统计
            stats = {
                'count': len(numeric_values),
                'sum': sum(numeric_values),
                'mean': np.mean(numeric_values),
                'median': np.median(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values),
                'std': np.std(numeric_values),
                'var': np.var(numeric_values)
            }
        
        return stats
    
    async def _merge_custom(
        self,
        results: List[TaskResult],
        rule: MergeRule,
        custom_aggregation: Optional[Callable] = None
    ) -> Any:
        """自定义合并"""
        if custom_aggregation:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, custom_aggregation, [r.result for r in results]
            )
        
        if rule.custom_function:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, rule.custom_function, [r.result for r in results]
            )
        
        # 默认返回第一个结果
        return results[0].result if results else None
    
    def get_merge_statistics(self) -> dict:
        """获取合并统计信息"""
        return self.stats.copy()
    
    def get_merged_result(self, cache_key: str) -> Optional[Any]:
        """获取已合并的结果"""
        return self.merged_results.get(cache_key)
    
    def clear_results(self, task_ids: List[str] = None):
        """清理结果"""
        with self._lock:
            if task_ids:
                for task_id in task_ids:
                    self.results.pop(task_id, None)
            else:
                self.results.clear()
            
            logger.info("结果已清理")


class ResultAggregator(ResultMerger):
    """结果聚合器，继承自ResultMerger"""
    
    def __init__(self, max_workers: int = 5):
        super().__init__(max_workers)
        self.aggregation_rules: Dict[str, Callable] = {}
        self.grouped_results: Dict[str, List[TaskResult]] = defaultdict(list)
        
        logger.info("结果聚合器初始化完成")
    
    def register_aggregation_rule(self, name: str, rule_func: Callable):
        """注册聚合规则"""
        self.aggregation_rules[name] = rule_func
        logger.info(f"聚合规则已注册: {name}")
    
    async def aggregate_by_group(
        self,
        group_key: str,
        task_ids: List[str],
        aggregation_rule: str = "default"
    ) -> Dict[str, Any]:
        """按组聚合结果"""
        # 按组分类结果
        grouped = defaultdict(list)
        
        for task_id in task_ids:
            if task_id in self.results:
                result = self.results[task_id]
                # 假设结果中包含分组键
                if isinstance(result.result, dict) and group_key in result.result:
                    group_value = result.result[group_key]
                    grouped[group_value].append(result)
        
        # 对每个组执行聚合
        aggregated = {}
        
        for group_value, group_results in grouped.items():
            if aggregation_rule in self.aggregation_rules:
                # 使用自定义聚合规则
                aggregated[group_value] = await self.aggregation_rules[aggregation_rule](group_results)
            else:
                # 使用默认聚合
                aggregated[group_value] = await self.merge_results(
                    [r.task_id for r in group_results],
                    aggregation_rule
                )
        
        return aggregated
    
    async def aggregate_with_weights(
        self,
        task_ids: List[str],
        weights: Dict[str, float],
        aggregation_rule: str = "weighted_average"
    ) -> Any:
        """带权重的聚合"""
        weighted_results = []
        
        for task_id in task_ids:
            if task_id in self.results and task_id in weights:
                result = self.results[task_id]
                weighted_results.append((result.result, weights[task_id]))
        
        if not weighted_results:
            return None
        
        if aggregation_rule == "weighted_average":
            # 计算加权平均
            total_weight = sum(weight for _, weight in weighted_results)
            if total_weight == 0:
                return None
            
            weighted_sum = sum(value * weight for value, weight in weighted_results)
            return weighted_sum / total_weight
        
        elif aggregation_rule == "weighted_sum":
            # 计算加权和
            return sum(value * weight for value, weight in weighted_results)
        
        else:
            # 默认使用第一个结果
            return weighted_results[0][0]
    
    async def aggregate_temporal(
        self,
        task_ids: List[str],
        time_window: float = 60.0,
        aggregation_rule: str = "time_based_average"
    ) -> Dict[str, List[Any]]:
        """时间序列聚合"""
        # 按时间窗口分组
        time_groups = defaultdict(list)
        
        for task_id in task_ids:
            if task_id in self.results:
                result = self.results[task_id]
                # 计算时间窗口
                window_start = int(result.timestamp // time_window) * time_window
                time_groups[window_start].append(result.result)
        
        # 对每个时间窗口执行聚合
        aggregated = {}
        
        for window_start, values in time_groups.items():
            if aggregation_rule == "time_based_average":
                if values and all(isinstance(v, (int, float)) for v in values):
                    aggregated[window_start] = np.mean(values)
                else:
                    aggregated[window_start] = values
            elif aggregation_rule == "time_based_sum":
                if values and all(isinstance(v, (int, float)) for v in values):
                    aggregated[window_start] = sum(values)
                else:
                    aggregated[window_start] = values
            else:
                aggregated[window_start] = values
        
        return aggregated
    
    def export_results(self, filepath: str, format: str = "json"):
        """导出结果"""
        export_data = {
            'results': {},
            'merged_results': self.merged_results,
            'statistics': self.stats,
            'export_time': time.time()
        }
        
        for task_id, result in self.results.items():
            export_data['results'][task_id] = {
                'task_id': result.task_id,
                'result': result.result,
                'status': result.status,
                'execution_time': result.execution_time,
                'metadata': result.metadata,
                'timestamp': result.timestamp,
                'error': str(result.error) if result.error else None,
                'warnings': result.warnings
            }
        
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        logger.info(f"结果已导出到: {filepath}")


# 内置合并规则
def create_default_merge_rules(merger: ResultMerger):
    """创建默认合并规则"""
    
    # 列表合并规则
    merger.register_merge_rule(
        "list_first",
        MergeRule(
            strategy=MergeStrategy.FIRST,
            aggregation_type=AggregationType.LIST,
            priority=1
        )
    )
    
    merger.register_merge_rule(
        "list_concat",
        MergeRule(
            strategy=MergeStrategy.CONCAT,
            aggregation_type=AggregationType.LIST,
            priority=2
        )
    )
    
    merger.register_merge_rule(
        "list_union",
        MergeRule(
            strategy=MergeStrategy.UNION,
            aggregation_type=AggregationType.LIST,
            priority=3
        )
    )
    
    # 字典合并规则
    merger.register_merge_rule(
        "dict_union",
        MergeRule(
            strategy=MergeStrategy.UNION,
            aggregation_type=AggregationType.DICT,
            priority=1
        )
    )
    
    merger.register_merge_rule(
        "dict_intersection",
        MergeRule(
            strategy=MergeStrategy.INTERSECTION,
            aggregation_type=AggregationType.DICT,
            priority=2
        )
    )
    
    # 统计合并规则
    merger.register_merge_rule(
        "stats_sum",
        MergeRule(
            strategy=MergeStrategy.SUM,
            aggregation_type=AggregationType.STATISTICS,
            priority=1
        )
    )
    
    merger.register_merge_rule(
        "stats_average",
        MergeRule(
            strategy=MergeStrategy.AVERAGE,
            aggregation_type=AggregationType.STATISTICS,
            priority=2
        )
    )
    
    merger.register_merge_rule(
        "stats_complete",
        MergeRule(
            strategy=MergeStrategy.CUSTOM,
            aggregation_type=AggregationType.STATISTICS,
            priority=3
        )
    )


# 自定义聚合函数示例
def weighted_average_aggregation(values: List[tuple]) -> float:
    """加权平均聚合函数"""
    total_weight = sum(weight for _, weight in values)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(value * weight for value, weight in values)
    return weighted_sum / total_weight


def confidence_interval_aggregation(values: List[float], confidence: float = 0.95) -> dict:
    """置信区间聚合函数"""
    if not values:
        return {}
    
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    
    # 计算置信区间
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * std / np.sqrt(n)
    
    return {
        'mean': mean,
        'std': std,
        'count': n,
        'confidence_interval': (mean - margin_error, mean + margin_error),
        'confidence_level': confidence
    }


# 示例用法
async def example_usage():
    """示例用法"""
    # 创建结果合并器
    merger = ResultAggregator()
    
    # 创建默认合并规则
    create_default_merge_rules(merger)
    
    # 添加测试结果
    results = [
        TaskResult("task1", [1, 2, 3], "success", 1.0),
        TaskResult("task2", [4, 5, 6], "success", 1.2),
        TaskResult("task3", {"key1": "value1", "key2": "value2"}, "success", 0.8),
        TaskResult("task4", {"key1": "value3", "key3": "value4"}, "success", 1.1),
        TaskResult("task5", 85, "success", 0.9),
        TaskResult("task6", 92, "success", 1.0),
        TaskResult("task7", 78, "success", 1.1)
    ]
    
    merger.add_results_batch(results)
    
    # 测试不同的合并策略
    list_result = await merger.merge_results(["task1", "task2"], "list_concat")
    print("列表合并结果:", list_result)
    
    dict_result = await merger.merge_results(["task3", "task4"], "dict_union")
    print("字典合并结果:", dict_result)
    
    stats_result = await merger.merge_results(["task5", "task6", "task7"], "stats_complete")
    print("统计合并结果:", stats_result)
    
    # 获取统计信息
    stats = merger.get_merge_statistics()
    print("合并统计:", stats)
    
    # 导出结果
    merger.export_results("merge_results.json")
    
    print("结果合并示例完成")


if __name__ == "__main__":
    asyncio.run(example_usage())