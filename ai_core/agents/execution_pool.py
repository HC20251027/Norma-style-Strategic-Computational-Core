"""
任务执行池
负责管理工作线程、执行任务、负载均衡和性能优化
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import time
import uuid
import psutil
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import weakref
import json
from queue import Queue, Empty

# 配置日志
logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """工作线程状态"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class Worker:
    """工作线程"""
    id: str
    name: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_tasks: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 1
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    weight: float = 1.0
    capabilities: Set[str] = field(default_factory=set)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    @property
    def utilization(self) -> float:
        """获取工作线程利用率"""
        if self.max_concurrent_tasks == 0:
            return 0.0
        return len(self.current_tasks) / self.max_concurrent_tasks
    
    @property
    def success_rate(self) -> float:
        """获取成功率"""
        total_tasks = self.completed_tasks + self.failed_tasks
        if total_tasks == 0:
            return 1.0
        return self.completed_tasks / total_tasks
    
    @property
    def avg_execution_time(self) -> float:
        """获取平均执行时间"""
        if self.completed_tasks == 0:
            return 0.0
        return self.total_execution_time / self.completed_tasks


@dataclass
class TaskExecution:
    """任务执行信息"""
    task_id: str
    worker_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0


class ExecutionPool:
    """任务执行池"""
    
    def __init__(
        self,
        name: str,
        max_workers: int = 10,
        min_workers: int = 2,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
        enable_monitoring: bool = True
    ):
        self.name = name
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.load_balancing_strategy = load_balancing_strategy
        self.enable_monitoring = enable_monitoring
        
        # 工作线程管理
        self.workers: Dict[str, Worker] = {}
        self.available_workers: deque = deque()
        self.busy_workers: Set[str] = set()
        
        # 任务队列
        self.task_queue = deque()
        self.executing_tasks: Dict[str, TaskExecution] = {}
        self.completed_tasks: Dict[str, TaskExecution] = {}
        self.failed_tasks: Dict[str, TaskExecution] = {}
        
        # 执行控制
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self._lock = threading.RLock()
        self._shutdown_event = asyncio.Event()
        
        # 负载均衡
        self.round_robin_index = 0
        self.worker_performance: Dict[str, float] = defaultdict(float)
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0,
            'total_execution_time': 0.0,
            'queue_length': 0,
            'active_workers': 0,
            'worker_utilization': 0.0
        }
        
        # 监控
        self._monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 5.0
        
        # 回调函数
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        logger.info(f"执行池 {name} 初始化完成，最大工作线程数: {max_workers}")
    
    async def start(self):
        """启动执行池"""
        if self.running:
            logger.warning(f"执行池 {self.name} 已在运行")
            return
        
        self.running = True
        self._shutdown_event.clear()
        
        # 启动最小工作线程数
        await self._scale_workers(self.min_workers)
        
        # 启动调度循环
        asyncio.create_task(self._scheduler_loop())
        
        # 启动监控
        if self.enable_monitoring:
            asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"执行池 {self.name} 已启动")
    
    async def stop(self):
        """停止执行池"""
        if not self.running:
            return
        
        self.running = False
        self._shutdown_event.set()
        
        # 取消所有运行中的任务
        for task_execution in self.executing_tasks.values():
            if hasattr(task_execution, 'future') and task_execution.future:
                task_execution.future.cancel()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        # 等待监控任务完成
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"执行池 {self.name} 已停止")
    
    async def _scale_workers(self, target_count: int):
        """调整工作线程数量"""
        current_count = len(self.workers)
        
        if target_count > current_count:
            # 增加工作线程
            for i in range(target_count - current_count):
                await self._add_worker()
        elif target_count < current_count:
            # 减少工作线程
            workers_to_remove = current_count - target_count
            await self._remove_workers(workers_to_remove)
    
    async def _add_worker(self) -> str:
        """添加工作线程"""
        worker_id = str(uuid.uuid4())
        worker_name = f"Worker-{worker_id[:8]}"
        
        worker = Worker(
            id=worker_id,
            name=worker_name,
            max_concurrent_tasks=1,  # 默认单线程
            capabilities={"general"}
        )
        
        self.workers[worker_id] = worker
        self.available_workers.append(worker_id)
        
        logger.debug(f"工作线程已添加: {worker_name}")
        return worker_id
    
    async def _remove_workers(self, count: int):
        """移除工作线程"""
        removed = 0
        
        while removed < count and self.available_workers:
            worker_id = self.available_workers.popleft()
            worker = self.workers[worker_id]
            
            if len(worker.current_tasks) == 0:
                worker.status = WorkerStatus.OFFLINE
                del self.workers[worker_id]
                removed += 1
                logger.debug(f"工作线程已移除: {worker.name}")
    
    async def submit_task(
        self,
        task_id: str,
        task_func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        timeout: Optional[float] = None,
        worker_requirements: dict = None
    ) -> str:
        """提交任务"""
        kwargs = kwargs or {}
        worker_requirements = worker_requirements or {}
        
        # 创建任务执行记录
        task_execution = TaskExecution(
            task_id=task_id,
            worker_id="",
            start_time=time.time()
        )
        
        self.executing_tasks[task_id] = task_execution
        self.stats['total_tasks'] += 1
        
        # 添加到队列
        task_info = {
            'execution': task_execution,
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'timeout': timeout,
            'requirements': worker_requirements
        }
        
        self.task_queue.append(task_info)
        self.stats['queue_length'] += 1
        
        logger.debug(f"任务已提交: {task_id}")
        return task_id
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                # 处理队列中的任务
                await self._process_queue()
                
                # 更新统计信息
                self._update_stats()
                
                # 动态调整工作线程数
                await self._auto_scale()
                
                # 等待一段时间
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"执行池调度错误: {e}")
                await asyncio.sleep(1)
    
    async def _process_queue(self):
        """处理任务队列"""
        while self.task_queue and self.available_workers:
            task_info = self.task_queue.popleft()
            self.stats['queue_length'] -= 1
            
            # 选择工作线程
            worker_id = await self._select_worker(task_info['requirements'])
            
            if worker_id:
                await self._execute_task_on_worker(worker_id, task_info)
            else:
                # 没有可用工作线程，重新放回队列
                self.task_queue.appendleft(task_info)
                self.stats['queue_length'] += 1
                break
    
    async def _select_worker(self, requirements: dict) -> Optional[str]:
        """选择工作线程"""
        if not self.available_workers:
            return None
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_worker_round_robin()
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_worker_least_connections()
        elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_worker_weighted_round_robin()
        elif self.load_balancing_strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._select_worker_resource_aware(requirements)
        elif self.load_balancing_strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._select_worker_performance_based()
        else:
            return self.available_workers[0]
    
    def _select_worker_round_robin(self) -> str:
        """轮询选择工作线程"""
        worker_id = self.available_workers[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % len(self.available_workers)
        return worker_id
    
    def _select_worker_least_connections(self) -> str:
        """最少连接选择工作线程"""
        best_worker = None
        min_connections = float('inf')
        
        for worker_id in self.available_workers:
            worker = self.workers[worker_id]
            connections = len(worker.current_tasks)
            
            if connections < min_connections:
                min_connections = connections
                best_worker = worker_id
        
        return best_worker
    
    def _select_worker_weighted_round_robin(self) -> str:
        """加权轮询选择工作线程"""
        total_weight = sum(self.workers[wid].weight for wid in self.available_workers)
        if total_weight == 0:
            return self.available_workers[0]
        
        import random
        target = random.uniform(0, total_weight)
        current_weight = 0
        
        for worker_id in self.available_workers:
            current_weight += self.workers[worker_id].weight
            if current_weight >= target:
                return worker_id
        
        return self.available_workers[0]
    
    def _select_worker_resource_aware(self, requirements: dict) -> str:
        """资源感知选择工作线程"""
        best_worker = None
        best_score = -1
        
        for worker_id in self.available_workers:
            worker = self.workers[worker_id]
            
            # 计算资源匹配分数
            score = 0
            
            # CPU使用率分数
            cpu_score = max(0, 1.0 - worker.cpu_usage)
            score += cpu_score * 0.3
            
            # 内存使用率分数
            memory_score = max(0, 1.0 - worker.memory_usage)
            score += memory_score * 0.3
            
            # 利用率分数
            utilization_score = max(0, 1.0 - worker.utilization)
            score += utilization_score * 0.4
            
            # 能力匹配加分
            required_capabilities = requirements.get('capabilities', set())
            if required_capabilities.issubset(worker.capabilities):
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_worker = worker_id
        
        return best_worker
    
    def _select_worker_performance_based(self) -> str:
        """基于性能选择工作线程"""
        best_worker = None
        best_performance = -1
        
        for worker_id in self.available_workers:
            worker = self.workers[worker_id]
            
            # 综合性能分数
            performance = 0
            
            # 成功率权重
            performance += worker.success_rate * 0.4
            
            # 平均响应时间权重（越短越好）
            if worker.avg_execution_time > 0:
                time_score = max(0, 1.0 / (1.0 + worker.avg_execution_time))
                performance += time_score * 0.3
            
            # 利用率权重
            utilization_score = max(0, 1.0 - worker.utilization)
            performance += utilization_score * 0.3
            
            if performance > best_performance:
                best_performance = performance
                best_worker = worker_id
        
        return best_worker
    
    async def _execute_task_on_worker(self, worker_id: str, task_info: dict):
        """在工作线程上执行任务"""
        worker = self.workers[worker_id]
        task_execution = task_info['execution']
        
        # 更新工作线程状态
        worker.status = WorkerStatus.BUSY
        worker.current_tasks.add(task_execution.task_id)
        self.available_workers.remove(worker_id)
        self.busy_workers.add(worker_id)
        
        # 创建执行future
        loop = asyncio.get_event_loop()
        
        async def execute_with_monitoring():
            try:
                # 执行任务
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        lambda: task_info['func'](*task_info['args'], **task_info['kwargs'])
                    ),
                    timeout=task_info['timeout']
                )
                
                # 任务成功完成
                task_execution.result = result
                task_execution.status = "completed"
                task_execution.end_time = time.time()
                
                # 更新工作线程统计
                worker.completed_tasks += 1
                worker.total_execution_time += (task_execution.end_time - task_execution.start_time)
                
                # 更新性能指标
                execution_time = task_execution.end_time - task_execution.start_time
                self.worker_performance[worker_id] = execution_time
                
                logger.debug(f"任务执行成功: {task_execution.task_id} 在 {worker.name}")
                
            except asyncio.TimeoutError:
                await self._handle_task_timeout(worker, task_execution)
            except Exception as e:
                await self._handle_task_error(worker, task_execution, e)
            finally:
                # 清理任务执行记录
                if task_execution.task_id in self.executing_tasks:
                    del self.executing_tasks[task_execution.task_id]
                
                # 更新工作线程状态
                worker.current_tasks.discard(task_execution.task_id)
                worker.status = WorkerStatus.IDLE
                
                # 重新加入可用队列
                if worker.status == WorkerStatus.IDLE:
                    self.available_workers.append(worker_id)
                    self.busy_workers.discard(worker_id)
        
        # 启动执行
        task_execution.future = asyncio.create_task(execute_with_monitoring())
        
        # 保存执行记录
        task_execution.worker_id = worker_id
    
    async def _handle_task_timeout(self, worker: Worker, task_execution: TaskExecution):
        """处理任务超时"""
        task_execution.status = "timeout"
        task_execution.end_time = time.time()
        task_execution.error = TimeoutError("任务执行超时")
        
        worker.failed_tasks += 1
        
        logger.warning(f"任务超时: {task_execution.task_id} 在 {worker.name}")
    
    async def _handle_task_error(self, worker: Worker, task_execution: TaskExecution, error: Exception):
        """处理任务错误"""
        task_execution.status = "failed"
        task_execution.end_time = time.time()
        task_execution.error = error
        
        worker.failed_tasks += 1
        
        logger.error(f"任务执行失败: {task_execution.task_id} 在 {worker.name}, 错误: {error}")
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats['completed_tasks'] = len(self.completed_tasks)
        self.stats['failed_tasks'] = len(self.failed_tasks)
        self.stats['active_workers'] = len(self.workers)
        self.stats['queue_length'] = len(self.task_queue)
        
        # 计算平均执行时间
        if self.stats['completed_tasks'] > 0:
            total_time = sum(
                execution.end_time - execution.start_time
                for execution in self.completed_tasks.values()
                if execution.end_time
            )
            self.stats['avg_execution_time'] = total_time / self.stats['completed_tasks']
        
        # 计算工作线程利用率
        if self.workers:
            total_utilization = sum(worker.utilization for worker in self.workers.values())
            self.stats['worker_utilization'] = total_utilization / len(self.workers)
    
    async def _auto_scale(self):
        """自动扩缩容"""
        if len(self.workers) >= self.max_workers:
            return
        
        # 根据队列长度和利用率决定是否扩容
        queue_ratio = len(self.task_queue) / max(1, len(self.workers))
        avg_utilization = self.stats['worker_utilization']
        
        # 如果队列很长或利用率很高，考虑扩容
        if queue_ratio > 2.0 or avg_utilization > 0.8:
            await self._add_worker()
            logger.debug("执行池自动扩容")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                await self._update_worker_metrics()
                await self._check_worker_health()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"执行池监控错误: {e}")
                await asyncio.sleep(1)
    
    async def _update_worker_metrics(self):
        """更新工作线程指标"""
        for worker in self.workers.values():
            # 更新CPU和内存使用率
            worker.cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
            worker.memory_usage = psutil.virtual_memory().percent / 100.0
            
            # 更新响应时间
            if worker.completed_tasks > 0:
                worker.response_time = worker.avg_execution_time
            
            # 更新心跳时间
            worker.last_heartbeat = time.time()
    
    async def _check_worker_health(self):
        """检查工作线程健康状态"""
        current_time = time.time()
        
        # 检查心跳超时的工作线程
        offline_workers = []
        for worker_id, worker in self.workers.items():
            if current_time - worker.last_heartbeat > 30:  # 30秒超时
                offline_workers.append(worker_id)
        
        # 标记离线工作线程
        for worker_id in offline_workers:
            worker = self.workers[worker_id]
            worker.status = WorkerStatus.OFFLINE
            
            # 从队列中移除
            if worker_id in self.available_workers:
                self.available_workers.remove(worker_id)
            self.busy_workers.discard(worker_id)
            
            logger.warning(f"工作线程离线: {worker.name}")
    
    def get_worker_status(self, worker_id: str) -> Optional[dict]:
        """获取工作线程状态"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            return {
                'id': worker.id,
                'name': worker.name,
                'status': worker.status.value,
                'current_tasks': list(worker.current_tasks),
                'max_concurrent_tasks': worker.max_concurrent_tasks,
                'utilization': worker.utilization,
                'cpu_usage': worker.cpu_usage,
                'memory_usage': worker.memory_usage,
                'response_time': worker.response_time,
                'completed_tasks': worker.completed_tasks,
                'failed_tasks': worker.failed_tasks,
                'success_rate': worker.success_rate,
                'avg_execution_time': worker.avg_execution_time,
                'capabilities': list(worker.capabilities)
            }
        return None
    
    def get_statistics(self) -> dict:
        """获取执行池统计信息"""
        return self.stats.copy()
    
    def get_queue_status(self) -> dict:
        """获取队列状态"""
        return {
            'queue_length': len(self.task_queue),
            'executing_tasks': len(self.executing_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'available_workers': len(self.available_workers),
            'busy_workers': len(self.busy_workers),
            'total_workers': len(self.workers)
        }
    
    def add_worker_capability(self, worker_id: str, capability: str):
        """为工作线程添加能力"""
        if worker_id in self.workers:
            self.workers[worker_id].capabilities.add(capability)
    
    def set_worker_weight(self, worker_id: str, weight: float):
        """设置工作线程权重"""
        if worker_id in self.workers:
            self.workers[worker_id].weight = weight
    
    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy):
        """设置负载均衡策略"""
        self.load_balancing_strategy = strategy
        logger.info(f"负载均衡策略已设置为: {strategy.value}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.executing_tasks:
            execution = self.executing_tasks[task_id]
            
            # 取消future
            if hasattr(execution, 'future') and execution.future:
                execution.future.cancel()
            
            # 标记为取消
            execution.status = "cancelled"
            execution.end_time = time.time()
            
            # 清理执行记录
            del self.executing_tasks[task_id]
            
            logger.info(f"任务已取消: {task_id}")
            return True
        
        # 检查是否在队列中
        for i, task_info in enumerate(self.task_queue):
            if task_info['execution'].task_id == task_id:
                del self.task_queue[i]
                self.stats['queue_length'] -= 1
                logger.info(f"任务已从队列移除: {task_id}")
                return True
        
        return False
    
    def save_configuration(self, filepath: str):
        """保存配置"""
        config = {
            'name': self.name,
            'max_workers': self.max_workers,
            'min_workers': self.min_workers,
            'load_balancing_strategy': self.load_balancing_strategy.value,
            'workers': {}
        }
        
        for worker_id, worker in self.workers.items():
            config['workers'][worker_id] = {
                'name': worker.name,
                'max_concurrent_tasks': worker.max_concurrent_tasks,
                'weight': worker.weight,
                'capabilities': list(worker.capabilities)
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"执行池配置已保存到: {filepath}")
    
    def load_configuration(self, filepath: str):
        """加载配置"""
        if not os.path.exists(filepath):
            logger.warning(f"配置文件不存在: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.max_workers = config.get('max_workers', self.max_workers)
        self.min_workers = config.get('min_workers', self.min_workers)
        
        strategy_name = config.get('load_balancing_strategy', 'least_connections')
        self.load_balancing_strategy = LoadBalancingStrategy(strategy_name)
        
        # 重新创建工作线程
        self.workers.clear()
        self.available_workers.clear()
        self.busy_workers.clear()
        
        for worker_id, worker_config in config.get('workers', {}).items():
            worker = Worker(
                id=worker_id,
                name=worker_config['name'],
                max_concurrent_tasks=worker_config.get('max_concurrent_tasks', 1),
                weight=worker_config.get('weight', 1.0),
                capabilities=set(worker_config.get('capabilities', ['general']))
            )
            self.workers[worker_id] = worker
            self.available_workers.append(worker_id)
        
        logger.info(f"执行池配置已从 {filepath} 加载")


# 辅助函数
def create_worker(
    name: str,
    max_concurrent_tasks: int = 1,
    capabilities: List[str] = None,
    weight: float = 1.0
) -> Worker:
    """创建工作线程"""
    worker_id = str(uuid.uuid4())
    capabilities = capabilities or ['general']
    
    return Worker(
        id=worker_id,
        name=name,
        max_concurrent_tasks=max_concurrent_tasks,
        capabilities=set(capabilities),
        weight=weight
    )


# 示例用法
async def example_usage():
    """示例用法"""
    # 创建执行池
    pool = ExecutionPool(
        name="MainPool",
        max_workers=5,
        min_workers=2,
        load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS
    )
    
    # 启动执行池
    await pool.start()
    
    # 定义任务函数
    def cpu_intensive_task(n):
        time.sleep(1)
        return sum(range(n))
    
    def io_task():
        time.sleep(2)
        return "I/O任务完成"
    
    # 提交任务
    await pool.submit_task("task1", cpu_intensive_task, args=(100000,))
    await pool.submit_task("task2", io_task)
    await pool.submit_task("task3", cpu_intensive_task, args=(50000,))
    
    # 等待任务完成
    await asyncio.sleep(5)
    
    # 获取统计信息
    stats = pool.get_statistics()
    print("执行池统计:", stats)
    
    queue_status = pool.get_queue_status()
    print("队列状态:", queue_status)
    
    # 停止执行池
    await pool.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())