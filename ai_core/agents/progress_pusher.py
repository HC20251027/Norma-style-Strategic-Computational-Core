"""
非阻塞等待系统进度推送器
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging

from .config import NonBlockingConfig
from .event_system import event_system, EventType, Event
from .task_manager import TaskManager, Task


@dataclass
class ProgressEvent:
    """进度事件"""
    task_id: str
    progress: float
    message: str
    timestamp: datetime
    estimated_time_remaining: Optional[float] = None
    speed: Optional[float] = None  # 每秒进度
    correlation_id: Optional[str] = None


class ProgressPredictor:
    """进度预测器"""
    
    def __init__(self):
        self.progress_history: Dict[str, List[tuple]] = {}  # task_id: [(timestamp, progress), ...]
        self.prediction_cache: Dict[str, float] = {}
    
    def record_progress(self, task_id: str, progress: float) -> None:
        """记录进度"""
        if task_id not in self.progress_history:
            self.progress_history[task_id] = []
        
        self.progress_history[task_id].append((time.time(), progress))
        
        # 保持历史记录在合理范围内
        if len(self.progress_history[task_id]) > 100:
            self.progress_history[task_id].pop(0)
    
    def predict_remaining_time(self, task_id: str, current_progress: float) -> Optional[float]:
        """预测剩余时间"""
        history = self.progress_history.get(task_id, [])
        if len(history) < 2:
            return None
        
        # 计算平均速度
        recent_history = history[-10:]  # 最近10个数据点
        total_time = recent_history[-1][0] - recent_history[0][0]
        total_progress = recent_history[-1][1] - recent_history[0][1]
        
        if total_time <= 0 or total_progress <= 0:
            return None
        
        speed = total_progress / total_time  # 每秒进度
        
        if speed <= 0:
            return None
        
        remaining_progress = 100 - current_progress
        remaining_time = remaining_progress / speed
        
        # 缓存预测结果
        self.prediction_cache[task_id] = remaining_time
        
        return remaining_time
    
    def get_average_speed(self, task_id: str) -> Optional[float]:
        """获取平均速度"""
        history = self.progress_history.get(task_id, [])
        if len(history) < 2:
            return None
        
        recent_history = history[-20:]  # 最近20个数据点
        total_time = recent_history[-1][0] - recent_history[0][0]
        total_progress = recent_history[-1][1] - recent_history[0][1]
        
        if total_time <= 0:
            return None
        
        return total_progress / total_time


class ProgressPusher:
    """进度推送器"""
    
    def __init__(self, task_manager: TaskManager, config: Optional[NonBlockingConfig] = None):
        self.task_manager = task_manager
        self.config = config or NonBlockingConfig()
        self.predictor = ProgressPredictor()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.push_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
    async def start(self) -> None:
        """启动进度推送器"""
        if self.running:
            return
        
        self.running = True
        logging.info("ProgressPusher started")
    
    async def stop(self) -> None:
        """停止进度推送器"""
        if not self.running:
            return
        
        self.running = False
        
        # 取消所有推送任务
        for task in self.push_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.push_tasks.clear()
        logging.info("ProgressPusher stopped")
    
    def subscribe(self, task_id: str, callback: Callable[[ProgressEvent], None]) -> str:
        """订阅任务进度"""
        if task_id not in self.subscribers:
            self.subscribers[task_id] = []
        
        subscriber_id = f"{task_id}:{len(self.subscribers[task_id])}"
        self.subscribers[task_id].append(callback)
        
        # 如果任务正在运行，开始推送进度
        task = self.task_manager.get_task(task_id)
        if task and task.status.value in ['running', 'queued']:
            asyncio.create_task(self._start_push_task(task_id))
        
        return subscriber_id
    
    def unsubscribe(self, task_id: str, subscriber_id: str) -> bool:
        """取消订阅"""
        if task_id not in self.subscribers:
            return False
        
        try:
            index = int(subscriber_id.split(':')[1])
            if 0 <= index < len(self.subscribers[task_id]):
                self.subscribers[task_id].pop(index)
                return True
        except (IndexError, ValueError):
            pass
        
        return False
    
    async def _start_push_task(self, task_id: str) -> None:
        """开始推送任务"""
        if task_id in self.push_tasks:
            return  # 已经有一个推送任务在运行
        
        task = asyncio.create_task(self._push_progress_loop(task_id))
        self.push_tasks[task_id] = task
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self.push_tasks.pop(task_id, None)
    
    async def _push_progress_loop(self, task_id: str) -> None:
        """进度推送循环"""
        last_progress = -1
        
        while self.running:
            task = self.task_manager.get_task(task_id)
            if not task:
                break
            
            # 任务已完成或失败，停止推送
            if task.status.value in ['completed', 'failed', 'cancelled', 'timeout']:
                # 发送最终进度
                final_progress = 100.0 if task.status.value == 'completed' else task.progress
                await self._push_progress(task_id, final_progress, self._get_status_message(task))
                break
            
            # 检查进度是否有变化
            if abs(task.progress - last_progress) >= self.config.progress_update_threshold or task.progress >= 100:
                await self._push_progress(task_id, task.progress, self._get_progress_message(task))
                last_progress = task.progress
            
            await asyncio.sleep(self.config.progress_push_interval)
    
    async def _push_progress(self, task_id: str, progress: float, message: str) -> None:
        """推送进度"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        # 记录进度历史
        self.predictor.record_progress(task_id, progress)
        
        # 预测剩余时间
        estimated_time_remaining = None
        speed = None
        
        if self.config.enable_progress_prediction:
            estimated_time_remaining = self.predictor.predict_remaining_time(task_id, progress)
            speed = self.predictor.get_average_speed(task_id)
        
        # 创建进度事件
        progress_event = ProgressEvent(
            task_id=task_id,
            progress=progress,
            message=message,
            timestamp=datetime.now(timezone.utc),
            estimated_time_remaining=estimated_time_remaining,
            speed=speed,
            correlation_id=task.correlation_id
        )
        
        # 推送给所有订阅者
        subscribers = self.subscribers.get(task_id, [])
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress_event)
                else:
                    callback(progress_event)
            except Exception as e:
                logging.error(f"Progress callback error: {e}")
        
        # 发送事件
        await event_system.emit_progress_update(task_id, progress, message, task.correlation_id)
    
    def _get_progress_message(self, task: Task) -> str:
        """获取进度消息"""
        if task.progress >= 100:
            return "任务完成"
        elif task.progress >= 90:
            return "即将完成"
        elif task.progress >= 75:
            return "大部分工作已完成"
        elif task.progress >= 50:
            return "进行中..."
        elif task.progress >= 25:
            return "开始处理"
        elif task.progress > 0:
            return "初始化中"
        else:
            return "准备开始"
    
    def _get_status_message(self, task: Task) -> str:
        """获取状态消息"""
        status_messages = {
            "completed": "任务已完成",
            "failed": f"任务失败: {task.error}",
            "cancelled": "任务已取消",
            "timeout": "任务超时"
        }
        return status_messages.get(task.status.value, "任务状态未知")
    
    async def push_manual_progress(self, task_id: str, progress: float, message: str) -> None:
        """手动推送进度"""
        await self._push_progress(task_id, progress, message)
    
    def get_prediction(self, task_id: str) -> Optional[Dict[str, float]]:
        """获取预测信息"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return None
        
        current_progress = task.progress
        estimated_time_remaining = self.predictor.predict_remaining_time(task_id, current_progress)
        speed = self.predictor.get_average_speed(task_id)
        
        return {
            "current_progress": current_progress,
            "estimated_time_remaining": estimated_time_remaining,
            "average_speed": speed,
            "prediction_confidence": self._calculate_prediction_confidence(task_id)
        }
    
    def _calculate_prediction_confidence(self, task_id: str) -> float:
        """计算预测置信度"""
        history = self.predictor.progress_history.get(task_id, [])
        if len(history) < 5:
            return 0.0
        
        # 基于历史数据点的数量和一致性计算置信度
        confidence = min(len(history) / 20.0, 1.0)  # 最多20个数据点达到100%置信度
        
        # 检查进度趋势的一致性
        if len(history) >= 3:
            recent_progress = [h[1] for h in history[-5:]]
            # 检查是否单调递增（允许小的波动）
            consistent = all(
                recent_progress[i] <= recent_progress[i+1] + 5 
                for i in range(len(recent_progress)-1)
            )
            if consistent:
                confidence *= 1.2  # 增加置信度
            else:
                confidence *= 0.8  # 降低置信度
        
        return min(confidence, 1.0)
    
    def get_active_push_tasks(self) -> Dict[str, asyncio.Task]:
        """获取活跃的推送任务"""
        return self.push_tasks.copy()
    
    def get_subscriber_count(self, task_id: str) -> int:
        """获取订阅者数量"""
        return len(self.subscribers.get(task_id, []))