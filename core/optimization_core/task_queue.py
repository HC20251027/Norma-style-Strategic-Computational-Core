"""
任务队列管理

支持多种队列类型、优先级管理和批量操作
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import heapq
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class QueueType(Enum):
    """队列类型"""
    FIFO = "fifo"  # 先进先出
    LIFO = "lifo"  # 后进先出
    PRIORITY = "priority"  # 优先级队列
    DELAYED = "delayed"  # 延迟队列
    BATCH = "batch"  # 批量队列


class QueueStatus(Enum):
    """队列状态"""
    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPED = "stopped"


@dataclass
class QueueItem:
    """队列项"""
    id: str
    data: Any
    priority: int = 0
    delay: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        if hasattr(other, 'priority'):
            return self.priority < other.priority
        return False


@dataclass
class BatchItem:
    """批量项"""
    batch_id: str
    items: List[QueueItem]
    batch_size: int
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskQueue:
    """任务队列管理器"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues: Dict[QueueType, Any] = {
            QueueType.FIFO: deque(),
            QueueType.LIFO: deque(),
            QueueType.PRIORITY: [],
            QueueType.DELAYED: [],
            QueueType.BATCH: deque()
        }
        
        self.queue_status: Dict[QueueType, QueueStatus] = {
            queue_type: QueueStatus.ACTIVE for queue_type in QueueType
        }
        
        self.metrics = {
            QueueType.FIFO: {'enqueued': 0, 'dequeued': 0, 'current_size': 0},
            QueueType.LIFO: {'enqueued': 0, 'dequeued': 0, 'current_size': 0},
            QueueType.PRIORITY: {'enqueued': 0, 'dequeued': 0, 'current_size': 0},
            QueueType.DELAYED: {'enqueued': 0, 'dequeued': 0, 'current_size': 0},
            QueueType.BATCH: {'enqueued': 0, 'dequeued': 0, 'current_size': 0}
        }
        
        self.delay_handlers: Dict[str, Callable] = {}
        self.batch_handlers: Dict[str, Callable] = {}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        self._lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 启动延迟队列处理器
        self._delay_processor_task = None
        self._running = False
    
    async def start(self):
        """启动队列管理器"""
        self._running = True
        self._delay_processor_task = asyncio.create_task(self._process_delayed_queue())
        logger.info("任务队列管理器已启动")
    
    async def stop(self):
        """停止队列管理器"""
        self._running = False
        if self._delay_processor_task:
            self._delay_processor_task.cancel()
            try:
                await self._delay_processor_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)
        logger.info("任务队列管理器已停止")
    
    async def enqueue(self, queue_type: QueueType, item: QueueItem) -> bool:
        """入队"""
        async with self._lock:
            if self.queue_status[queue_type] == QueueStatus.STOPPED:
                return False
            
            if self.metrics[queue_type]['current_size'] >= self.max_size:
                logger.warning(f"队列 {queue_type.value} 已满")
                return False
            
            try:
                if queue_type == QueueType.FIFO:
                    self.queues[QueueType.FIFO].append(item)
                elif queue_type == QueueType.LIFO:
                    self.queues[QueueType.LIFO].appendleft(item)
                elif queue_type == QueueType.PRIORITY:
                    heapq.heappush(self.queues[QueueType.PRIORITY], item)
                elif queue_type == QueueType.DELAYED:
                    execute_time = time.time() + item.delay
                    heapq.heappush(self.queues[QueueType.DELAYED], (execute_time, item))
                elif queue_type == QueueType.BATCH:
                    self.queues[QueueType.BATCH].append(item)
                
                self.metrics[queue_type]['enqueued'] += 1
                self.metrics[queue_type]['current_size'] += 1
                
                logger.debug(f"项目已入队 {queue_type.value}: {item.id}")
                return True
                
            except Exception as e:
                logger.error(f"入队失败: {e}")
                return False
    
    async def dequeue(self, queue_type: QueueType, timeout: Optional[float] = None) -> Optional[QueueItem]:
        """出队"""
        async with self._lock:
            if self.queue_status[queue_type] != QueueType.ACTIVE:
                return None
            
            try:
                item = None
                if queue_type == QueueType.FIFO:
                    if self.queues[QueueType.FIFO]:
                        item = self.queues[QueueType.FIFO].popleft()
                elif queue_type == QueueType.LIFO:
                    if self.queues[QueueType.LIFO]:
                        item = self.queues[QueueType.LIFO].pop()
                elif queue_type == QueueType.PRIORITY:
                    if self.queues[QueueType.PRIORITY]:
                        item = heapq.heappop(self.queues[QueueType.PRIORITY])
                elif queue_type == QueueType.BATCH:
                    if self.queues[QueueType.BATCH]:
                        item = self.queues[QueueType.BATCH].popleft()
                
                if item:
                    self.metrics[queue_type]['dequeued'] += 1
                    self.metrics[queue_type]['current_size'] -= 1
                    
                    # 触发回调
                    await self._trigger_callbacks('dequeue', item)
                    
                    logger.debug(f"项目已出队 {queue_type.value}: {item.id}")
                    return item
                
                return None
                
            except Exception as e:
                logger.error(f"出队失败: {e}")
                return None
    
    async def batch_enqueue(self, queue_type: QueueType, items: List[QueueItem], 
                          batch_size: int = 10) -> str:
        """批量入队"""
        batch_id = f"batch_{int(time.time() * 1000)}"
        
        try:
            # 创建批量项
            batch_item = BatchItem(
                batch_id=batch_id,
                items=items,
                batch_size=batch_size,
                metadata={'queue_type': queue_type}
            )
            
            # 入队到批量队列
            success = await self.enqueue(QueueType.BATCH, batch_item)
            
            if success:
                logger.info(f"批量入队成功: {batch_id}, 项目数: {len(items)}")
                return batch_id
            else:
                logger.error(f"批量入队失败: {batch_id}")
                return ""
                
        except Exception as e:
            logger.error(f"批量入队异常: {e}")
            return ""
    
    async def batch_dequeue(self, queue_type: QueueType, batch_size: int = 10) -> Optional[BatchItem]:
        """批量出队"""
        try:
            items = []
            
            # 从指定队列批量出队
            for _ in range(batch_size):
                item = await self.dequeue(queue_type)
                if item:
                    items.append(item)
                else:
                    break
            
            if items:
                batch_item = BatchItem(
                    batch_id=f"batch_{int(time.time() * 1000)}",
                    items=items,
                    batch_size=len(items),
                    metadata={'source_queue': queue_type}
                )
                
                logger.debug(f"批量出队成功: {len(items)} 项")
                return batch_item
            
            return None
            
        except Exception as e:
            logger.error(f"批量出队异常: {e}")
            return None
    
    async def peek(self, queue_type: QueueType) -> Optional[QueueItem]:
        """查看队首元素"""
        async with self._lock:
            try:
                if queue_type == QueueType.FIFO:
                    return self.queues[QueueType.FIFO][0] if self.queues[QueueType.FIFO] else None
                elif queue_type == QueueType.LIFO:
                    return self.queues[QueueType.LIFO][0] if self.queues[QueueType.LIFO] else None
                elif queue_type == QueueType.PRIORITY:
                    return self.queues[QueueType.PRIORITY][0] if self.queues[QueueType.PRIORITY] else None
                elif queue_type == QueueType.DELAYED:
                    if self.queues[QueueType.DELAYED]:
                        return self.queues[QueueType.DELAYED][0][1]
                    return None
                elif queue_type == QueueType.BATCH:
                    return self.queues[QueueType.BATCH][0] if self.queues[QueueType.BATCH] else None
                return None
            except Exception as e:
                logger.error(f"查看队首元素失败: {e}")
                return None
    
    async def size(self, queue_type: QueueType) -> int:
        """获取队列大小"""
        return self.metrics[queue_type]['current_size']
    
    async def is_empty(self, queue_type: QueueType) -> bool:
        """检查队列是否为空"""
        return await self.size(queue_type) == 0
    
    async def is_full(self, queue_type: QueueType) -> bool:
        """检查队列是否已满"""
        return await self.size(queue_type) >= self.max_size
    
    async def clear(self, queue_type: QueueType):
        """清空队列"""
        async with self._lock:
            try:
                if queue_type in self.queues:
                    self.queues[queue_type].clear()
                    self.metrics[queue_type]['current_size'] = 0
                    logger.info(f"队列 {queue_type.value} 已清空")
            except Exception as e:
                logger.error(f"清空队列失败: {e}")
    
    async def pause_queue(self, queue_type: QueueType):
        """暂停队列"""
        self.queue_status[queue_type] = QueueStatus.PAUSED
        logger.info(f"队列 {queue_type.value} 已暂停")
    
    async def resume_queue(self, queue_type: QueueType):
        """恢复队列"""
        self.queue_status[queue_type] = QueueStatus.ACTIVE
        logger.info(f"队列 {queue_type.value} 已恢复")
    
    async def drain_queue(self, queue_type: QueueType) -> List[QueueItem]:
        """排空队列"""
        items = []
        async with self._lock:
            try:
                while not await self.is_empty(queue_type):
                    item = await self.dequeue(queue_type)
                    if item:
                        items.append(item)
                
                logger.info(f"队列 {queue_type.value} 已排空，获得 {len(items)} 项")
                return items
                
            except Exception as e:
                logger.error(f"排空队列失败: {e}")
                return items
    
    async def _process_delayed_queue(self):
        """处理延迟队列"""
        while self._running:
            try:
                await asyncio.sleep(0.1)  # 避免CPU占用过高
                
                current_time = time.time()
                delayed_items = []
                
                # 检查是否有延迟项到期
                while (self.queues[QueueType.DELAYED] and 
                       self.queues[QueueType.DELAYED][0][0] <= current_time):
                    
                    execute_time, item = heapq.heappop(self.queues[QueueType.DELAYED])
                    delayed_items.append(item)
                    
                    # 更新指标
                    self.metrics[QueueType.DELAYED]['dequeued'] += 1
                    self.metrics[QueueType.DELAYED]['current_size'] -= 1
                
                # 将到期的延迟项移动到FIFO队列
                for item in delayed_items:
                    await self.enqueue(QueueType.FIFO, item)
                    await self._trigger_callbacks('delay_expired', item)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"延迟队列处理异常: {e}")
    
    async def _trigger_callbacks(self, event_type: str, item: QueueItem):
        """触发回调函数"""
        try:
            callbacks = self.callbacks.get(event_type, [])
            for callback in callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(item)
                else:
                    callback(item)
        except Exception as e:
            logger.error(f"触发回调失败: {e}")
    
    def add_callback(self, event_type: str, callback: Callable):
        """添加回调函数"""
        self.callbacks[event_type].append(callback)
        logger.debug(f"已添加回调: {event_type}")
    
    def remove_callback(self, event_type: str, callback: Callable):
        """移除回调函数"""
        if callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            logger.debug(f"已移除回调: {event_type}")
    
    def get_metrics(self, queue_type: Optional[QueueType] = None) -> Dict[str, Any]:
        """获取队列指标"""
        if queue_type:
            return {
                'queue_type': queue_type.value,
                'status': self.queue_status[queue_type].value,
                'metrics': self.metrics[queue_type].copy()
            }
        else:
            return {
                'queues': {
                    qt.value: {
                        'status': self.queue_status[qt].value,
                        'metrics': self.metrics[qt].copy()
                    }
                    for qt in QueueType
                },
                'total_queues': len(QueueType),
                'max_size': self.max_size
            }
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        stats = {}
        for queue_type in QueueType:
            stats[queue_type.value] = {
                'size': await self.size(queue_type),
                'status': self.queue_status[queue_type].value,
                'metrics': self.metrics[queue_type].copy()
            }
        return stats