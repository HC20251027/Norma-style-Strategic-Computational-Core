"""
优先级调度管理器

支持动态优先级调整、优先级继承和抢占式调度
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import heapq
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class PriorityLevel(Enum):
    """优先级等级"""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    CRITICAL = 5
    EMERGENCY = 6


class SchedulingMode(Enum):
    """调度模式"""
    PREEMPTIVE = "preemptive"  # 抢占式
    COOPERATIVE = "cooperative"  # 协作式
    ROUND_ROBIN = "round_robin"  # 轮询
    FAIR_SHARE = "fair_share"  # 公平分享


@dataclass
class PriorityTask:
    """优先级任务"""
    id: str
    priority: PriorityLevel
    base_priority: PriorityLevel
    dynamic_priority: PriorityLevel
    created_at: float
    last_accessed: float
    execution_time: float = 0.0
    cpu_time: float = 0.0
    memory_usage: float = 0.0
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_preemptible: bool = True
    aging_factor: float = 1.0
    
    def __lt__(self, other):
        # 优先级队列比较：优先级越高越先执行
        if hasattr(other, 'dynamic_priority'):
            return self.dynamic_priority.value > other.dynamic_priority.value
        return False


@dataclass
class PriorityRule:
    """优先级规则"""
    rule_id: str
    name: str
    condition: str  # 条件表达式
    priority_adjustment: int
    is_active: bool = True
    created_at: float = field(default_factory=time.time)


class PriorityManager:
    """优先级调度管理器"""
    
    def __init__(self, max_tasks: int = 1000):
        self.max_tasks = max_tasks
        
        # 任务管理
        self.tasks: Dict[str, PriorityTask] = {}
        self.ready_queue: List[PriorityTask] = []
        self.running_tasks: Dict[str, PriorityTask] = {}
        self.completed_tasks: List[str] = []
        
        # 优先级规则
        self.priority_rules: Dict[str, PriorityRule] = {}
        self.rule_evaluators: Dict[str, Any] = {}
        
        # 调度配置
        self.scheduling_mode = SchedulingMode.PREEMPTIVE
        self.time_slice = 0.1  # 时间片（秒）
        self.aging_interval = 60.0  # 老化间隔（秒）
        self.max_priority_boost = 2  # 最大优先级提升
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'preemptions': 0,
            'priority_boosts': 0,
            'rule_applications': 0,
            'average_wait_time': 0.0,
            'average_response_time': 0.0
        }
        
        self._lock = asyncio.Lock()
        self._aging_task = None
        self._running = False
        
        # 负载监控
        self.load_history: List[float] = []
        self.load_window = 100  # 负载历史窗口大小
    
    async def start(self):
        """启动优先级管理器"""
        self._running = True
        self._aging_task = asyncio.create_task(self._aging_processor())
        logger.info("优先级调度管理器已启动")
    
    async def stop(self):
        """停止优先级管理器"""
        self._running = False
        if self._aging_task:
            self._aging_task.cancel()
            try:
                await self._aging_task
            except asyncio.CancelledError:
                pass
        logger.info("优先级调度管理器已停止")
    
    async def register_task(self, task_id: str, base_priority: PriorityLevel,
                          is_preemptible: bool = True, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """注册任务"""
        async with self._lock:
            try:
                if len(self.tasks) >= self.max_tasks:
                    logger.warning(f"任务数量已达上限 {self.max_tasks}")
                    return False
                
                priority_task = PriorityTask(
                    id=task_id,
                    priority=base_priority,
                    base_priority=base_priority,
                    dynamic_priority=base_priority,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    is_preemptible=is_preemptible,
                    metadata=metadata or {}
                )
                
                self.tasks[task_id] = priority_task
                heapq.heappush(self.ready_queue, priority_task)
                
                self.stats['total_tasks'] += 1
                
                # 应用优先级规则
                await self._apply_priority_rules(priority_task)
                
                logger.info(f"任务 {task_id} 注册成功，优先级: {base_priority.name}")
                return True
                
            except Exception as e:
                logger.error(f"任务注册失败: {e}")
                return False
    
    async def unregister_task(self, task_id: str) -> bool:
        """注销任务"""
        async with self._lock:
            try:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    # 从就绪队列中移除
                    if task in self.ready_queue:
                        self.ready_queue.remove(task)
                        heapq.heapify(self.ready_queue)
                    
                    # 从运行队列中移除
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]
                    
                    # 移除依赖关系
                    for dep_id in task.dependencies:
                        if dep_id in self.tasks:
                            self.tasks[dep_id].dependents.discard(task_id)
                    
                    for dep_id in task.dependents:
                        if dep_id in self.tasks:
                            self.tasks[dep_id].dependencies.discard(task_id)
                    
                    del self.tasks[task_id]
                    self.completed_tasks.append(task_id)
                    
                    logger.info(f"任务 {task_id} 已注销")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"任务注销失败: {e}")
                return False
    
    async def set_task_priority(self, task_id: str, new_priority: PriorityLevel) -> bool:
        """设置任务优先级"""
        async with self._lock:
            try:
                if task_id not in self.tasks:
                    return False
                
                task = self.tasks[task_id]
                old_priority = task.dynamic_priority
                task.base_priority = new_priority
                
                # 重新计算动态优先级
                await self._recalculate_dynamic_priority(task)
                
                # 如果任务在就绪队列中，重新排序
                if task in self.ready_queue:
                    heapq.heapify(self.ready_queue)
                
                # 检查是否需要抢占
                if (self.scheduling_mode == SchedulingMode.PREEMPTIVE and 
                    task.is_preemptible and
                    new_priority.value > old_priority.value):
                    
                    await self._check_preemption(task_id)
                
                logger.info(f"任务 {task_id} 优先级已更新: {old_priority.name} -> {new_priority.name}")
                return True
                
            except Exception as e:
                logger.error(f"设置任务优先级失败: {e}")
                return False
    
    async def boost_task_priority(self, task_id: str, boost: int) -> bool:
        """提升任务优先级"""
        async with self._lock:
            try:
                if task_id not in self.tasks:
                    return False
                
                task = self.tasks[task_id]
                old_priority = task.dynamic_priority
                
                # 计算新的优先级
                new_level = min(
                    PriorityLevel.EMERGENCY.value,
                    max(0, task.dynamic_priority.value + boost)
                )
                
                task.dynamic_priority = PriorityLevel(new_level)
                task.aging_factor = max(1.0, task.aging_factor + 0.1)
                
                self.stats['priority_boosts'] += 1
                
                # 重新排序就绪队列
                if task in self.ready_queue:
                    heapq.heapify(self.ready_queue)
                
                logger.info(f"任务 {task_id} 优先级已提升: {old_priority.name} -> {task.dynamic_priority.name}")
                return True
                
            except Exception as e:
                logger.error(f"提升任务优先级失败: {e}")
                return False
    
    async def add_dependency(self, task_id: str, depends_on: str) -> bool:
        """添加任务依赖"""
        async with self._lock:
            try:
                if task_id not in self.tasks or depends_on not in self.tasks:
                    return False
                
                self.tasks[task_id].dependencies.add(depends_on)
                self.tasks[depends_on].dependents.add(task_id)
                
                # 如果依赖任务未完成，降低当前任务优先级
                if depends_on not in self.completed_tasks:
                    await self._adjust_dependency_priority(task_id)
                
                logger.info(f"添加依赖关系: {task_id} -> {depends_on}")
                return True
                
            except Exception as e:
                logger.error(f"添加任务依赖失败: {e}")
                return False
    
    async def remove_dependency(self, task_id: str, depends_on: str) -> bool:
        """移除任务依赖"""
        async with self._lock:
            try:
                if task_id not in self.tasks or depends_on not in self.tasks:
                    return False
                
                self.tasks[task_id].dependencies.discard(depends_on)
                self.tasks[depends_on].dependents.discard(task_id)
                
                # 重新计算优先级
                await self._recalculate_dynamic_priority(self.tasks[task_id])
                
                logger.info(f"移除依赖关系: {task_id} -> {depends_on}")
                return True
                
            except Exception as e:
                logger.error(f"移除任务依赖失败: {e}")
                return False
    
    async def get_next_task(self) -> Optional[PriorityTask]:
        """获取下一个要执行的任务"""
        async with self._lock:
            try:
                while self.ready_queue:
                    task = heapq.heappop(self.ready_queue)
                    
                    # 检查依赖是否满足
                    if await self._are_dependencies_satisfied(task):
                        # 移动到运行队列
                        self.running_tasks[task.id] = task
                        task.last_accessed = time.time()
                        
                        logger.debug(f"选择任务执行: {task.id}, 优先级: {task.dynamic_priority.name}")
                        return task
                
                return None
                
            except Exception as e:
                logger.error(f"获取下一个任务失败: {e}")
                return None
    
    async def complete_task(self, task_id: str, execution_time: float):
        """完成任务"""
        async with self._lock:
            try:
                if task_id in self.running_tasks:
                    task = self.running_tasks[task_id]
                    
                    # 更新统计
                    task.execution_time += execution_time
                    self.stats['average_response_time'] = (
                        (self.stats['average_response_time'] * 
                         (len(self.completed_tasks) - 1) + execution_time) /
                        len(self.completed_tasks)
                        if self.completed_tasks else execution_time
                    )
                    
                    # 移除运行队列
                    del self.running_tasks[task_id]
                    
                    # 通知依赖此任务的其他任务
                    await self._notify_dependents(task_id)
                    
                    logger.info(f"任务 {task_id} 已完成，执行时间: {execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"完成任务处理失败: {e}")
    
    async def _recalculate_dynamic_priority(self, task: PriorityTask):
        """重新计算动态优先级"""
        try:
            # 基础优先级
            dynamic = task.base_priority.value
            
            # 考虑依赖关系
            unsatisfied_deps = len([dep for dep in task.dependencies 
                                  if dep not in self.completed_tasks])
            if unsatisfied_deps > 0:
                dynamic -= min(unsatisfied_deps, 2)  # 最多降低2级
            
            # 考虑等待时间
            wait_time = time.time() - task.created_at
            if wait_time > 60:  # 等待超过1分钟
                dynamic += min(int(wait_time / 60), 2)  # 最多提升2级
            
            # 应用老化因子
            dynamic = int(dynamic * task.aging_factor)
            
            # 限制范围
            dynamic = max(PriorityLevel.LOWEST.value, 
                         min(PriorityLevel.EMERGENCY.value, dynamic))
            
            task.dynamic_priority = PriorityLevel(dynamic)
            
        except Exception as e:
            logger.error(f"重新计算动态优先级失败: {e}")
    
    async def _apply_priority_rules(self, task: PriorityTask):
        """应用优先级规则"""
        try:
            for rule in self.priority_rules.values():
                if not rule.is_active:
                    continue
                
                # 评估规则条件（简化实现）
                if await self._evaluate_rule_condition(rule, task):
                    old_priority = task.dynamic_priority.value
                    new_priority = max(0, old_priority + rule.priority_adjustment)
                    task.dynamic_priority = PriorityLevel(new_priority)
                    
                    self.stats['rule_applications'] += 1
                    logger.debug(f"应用优先级规则 {rule.name}: {task.id}")
            
        except Exception as e:
            logger.error(f"应用优先级规则失败: {e}")
    
    async def _evaluate_rule_condition(self, rule: PriorityRule, task: PriorityTask) -> bool:
        """评估规则条件（简化实现）"""
        # 这里应该实现更复杂的条件评估逻辑
        # 目前使用简单的字符串匹配
        try:
            condition = rule.condition.lower()
            
            if 'priority' in condition:
                return True
            elif 'metadata' in condition and 'type' in task.metadata:
                return task.metadata['type'] in condition
            else:
                return False
                
        except Exception:
            return False
    
    async def _check_preemption(self, task_id: str):
        """检查是否需要抢占"""
        try:
            if task_id not in self.tasks:
                return
            
            high_priority_task = self.tasks[task_id]
            
            # 查找可以被抢占的低优先级任务
            preemptable_tasks = [
                task for task in self.running_tasks.values()
                if (task.is_preemptible and 
                    task.dynamic_priority.value < high_priority_task.dynamic_priority.value)
            ]
            
            if preemptable_tasks:
                # 选择优先级最低的任务进行抢占
                victim = min(preemptable_tasks, key=lambda t: t.dynamic_priority.value)
                
                # 抢占
                await self._preempt_task(victim.id, high_priority_task.id)
                
        except Exception as e:
            logger.error(f"检查抢占失败: {e}")
    
    async def _preempt_task(self, victim_id: str, preemptor_id: str):
        """执行任务抢占"""
        try:
            if victim_id in self.running_tasks:
                victim = self.running_tasks[victim_id]
                
                # 将被抢占的任务放回就绪队列
                del self.running_tasks[victim_id]
                heapq.heappush(self.ready_queue, victim)
                
                self.stats['preemptions'] += 1
                
                logger.info(f"任务抢占: {preemptor_id} 抢占 {victim_id}")
                
        except Exception as e:
            logger.error(f"任务抢占失败: {e}")
    
    async def _are_dependencies_satisfied(self, task: PriorityTask) -> bool:
        """检查依赖是否满足"""
        try:
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    return False
            return True
        except Exception as e:
            logger.error(f"检查依赖失败: {e}")
            return False
    
    async def _adjust_dependency_priority(self, task_id: str):
        """调整依赖任务的优先级"""
        try:
            task = self.tasks[task_id]
            unsatisfied_deps = len([dep for dep in task.dependencies 
                                  if dep not in self.completed_tasks])
            
            if unsatisfied_deps > 0:
                # 降低优先级
                new_level = max(0, task.dynamic_priority.value - 1)
                task.dynamic_priority = PriorityLevel(new_level)
                
        except Exception as e:
            logger.error(f"调整依赖优先级失败: {e}")
    
    async def _notify_dependents(self, completed_task_id: str):
        """通知依赖此任务的其他任务"""
        try:
            if completed_task_id in self.tasks:
                dependents = self.tasks[completed_task_id].dependents.copy()
                
                for dependent_id in dependents:
                    if dependent_id in self.tasks:
                        dependent = self.tasks[dependent_id]
                        
                        # 重新计算依赖任务的优先级
                        await self._recalculate_dynamic_priority(dependent)
                        
                        # 如果依赖已满足，加入就绪队列
                        if await self._are_dependencies_satisfied(dependent):
                            if dependent not in self.ready_queue:
                                heapq.heappush(self.ready_queue, dependent)
                
        except Exception as e:
            logger.error(f"通知依赖任务失败: {e}")
    
    async def _aging_processor(self):
        """老化处理器"""
        while self._running:
            try:
                await asyncio.sleep(self.aging_interval)
                
                current_time = time.time()
                
                # 对就绪队列中的任务进行老化
                for task in self.ready_queue:
                    wait_time = current_time - task.last_accessed
                    
                    if wait_time > self.aging_interval:
                        # 提升优先级
                        old_priority = task.dynamic_priority.value
                        new_priority = min(PriorityLevel.EMERGENCY.value, old_priority + 1)
                        task.dynamic_priority = PriorityLevel(new_priority)
                        task.aging_factor = max(1.0, task.aging_factor * 1.1)
                
                # 重新排序就绪队列
                heapq.heapify(self.ready_queue)
                
                logger.debug("优先级老化处理完成")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"老化处理器异常: {e}")
    
    def add_priority_rule(self, rule: PriorityRule):
        """添加优先级规则"""
        self.priority_rules[rule.rule_id] = rule
        logger.info(f"添加优先级规则: {rule.name}")
    
    def remove_priority_rule(self, rule_id: str) -> bool:
        """移除优先级规则"""
        if rule_id in self.priority_rules:
            del self.priority_rules[rule_id]
            logger.info(f"移除优先级规则: {rule_id}")
            return True
        return False
    
    def set_scheduling_mode(self, mode: SchedulingMode):
        """设置调度模式"""
        self.scheduling_mode = mode
        logger.info(f"调度模式已设置为: {mode.value}")
    
    def get_priority_stats(self) -> Dict[str, Any]:
        """获取优先级统计信息"""
        return {
            'total_tasks': len(self.tasks),
            'ready_tasks': len(self.ready_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'scheduling_mode': self.scheduling_mode.value,
            'stats': self.stats.copy(),
            'priority_distribution': {
                level.name: len([t for t in self.tasks.values() 
                               if t.dynamic_priority == level])
                for level in PriorityLevel
            }
        }