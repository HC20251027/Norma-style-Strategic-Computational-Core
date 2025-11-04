"""
非阻塞等待系统用户体验优化器
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random
import logging

from .config import NonBlockingConfig
from .event_system import event_system, EventType
from .task_manager import TaskManager, Task, TaskStatus


class FeedbackType(Enum):
    """反馈类型"""
    PROGRESS = "progress"
    ESTIMATE = "estimate"
    WARNING = "warning"
    SUCCESS = "success"
    ERROR = "error"
    QUEUE_INFO = "queue_info"
    MOTIVATION = "motivation"


class MessageTone(Enum):
    """消息语调"""
    NEUTRAL = "neutral"
    ENCOURAGING = "encouraging"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    URGENT = "urgent"


@dataclass
class UXEvent:
    """用户体验事件"""
    event_type: FeedbackType
    message: str
    tone: MessageTone
    timestamp: datetime
    task_id: Optional[str] = None
    correlation_id: Optional[str] = None
    data: Dict[str, Any] = None
    priority: int = 1  # 1-5, 5为最高优先级
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "message": self.message,
            "tone": self.tone.value,
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
            "data": self.data or {},
            "priority": self.priority
        }


class MessageTemplates:
    """消息模板"""
    
    PROGRESS_MESSAGES = {
        0: [
            "正在准备开始...",
            "初始化中，请稍候",
            "正在加载必要资源",
            "任务即将开始"
        ],
        10: [
            "开始处理您的请求",
            "正在进行初步分析",
            "任务已启动",
            "开始执行核心逻辑"
        ],
        25: [
            "正在处理中 ({progress}%)",
            "进展顺利 ({progress}%)",
            "正在执行主要功能",
            "处理进度良好"
        ],
        50: [
            "已完成一半 ({progress}%)",
            "正在处理关键部分",
            "进展到重要阶段",
            "任务进行顺利"
        ],
        75: [
            "即将完成 ({progress}%)",
            "正在进行最后的处理",
            "大部分工作已完成",
            "接近完成"
        ],
        90: [
            "即将完成 ({progress}%)",
            "正在进行最终整理",
            "任务即将结束",
            "最后阶段处理中"
        ],
        100: [
            "任务完成！",
            "处理已完成",
            "所有工作已完成",
            "任务执行成功"
        ]
    }
    
    ESTIMATE_MESSAGES = [
        "预计还需 {time} 秒",
        "大约需要 {time} 秒完成",
        "预计剩余时间：{time} 秒",
        "大约 {time} 秒后完成"
    ]
    
    QUEUE_MESSAGES = [
        "您前面还有 {count} 个任务",
        "当前队列位置：{position}",
        "预计等待时间：{time} 分钟",
        "队列中共有 {count} 个任务"
    ]
    
    MOTIVATION_MESSAGES = [
        "感谢您的耐心等待",
        "我们正在努力为您处理",
        "您的任务正在优先处理中",
        "请稍等，马上就好",
        "正在为您精心处理",
        "感谢您的理解与支持"
    ]
    
    ENCOURAGEMENT_MESSAGES = [
        "进展顺利，继续保持",
        "任务执行良好",
        "一切都在按计划进行",
        "处理效果很好",
        "进展超出预期"
    ]


class UXOptimizer:
    """用户体验优化器"""
    
    def __init__(self, task_manager: TaskManager, config: Optional[NonBlockingConfig] = None):
        self.task_manager = task_manager
        self.config = config or NonBlockingConfig()
        self.templates = MessageTemplates()
        self.message_history: Dict[str, List[UXEvent]] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.motivation_enabled = True
        self.last_motivation_time: Dict[str, datetime] = {}
        
    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """设置用户偏好"""
        self.user_preferences[user_id] = {
            "message_frequency": preferences.get("message_frequency", "normal"),  # low, normal, high
            "tone": preferences.get("tone", MessageTone.FRIENDLY),
            "enable_motivation": preferences.get("enable_motivation", True),
            "language": preferences.get("language", "zh"),
            "detailed_progress": preferences.get("detailed_progress", True)
        }
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """获取用户偏好"""
        return self.user_preferences.get(user_id, {
            "message_frequency": "normal",
            "tone": MessageTone.FRIENDLY,
            "enable_motivation": True,
            "language": "zh",
            "detailed_progress": True
        })
    
    async def send_progress_update(self, task_id: str, progress: float, message: str = None, correlation_id: str = None) -> None:
        """发送进度更新"""
        task = self.task_manager.get_task(task_id)
        if not task:
            return
        
        # 生成进度消息
        if not message:
            message = self._generate_progress_message(progress)
        
        # 获取用户偏好
        user_prefs = self.get_user_preferences(correlation_id or "")
        
        # 检查是否应该发送消息
        if not self._should_send_message(task_id, progress, user_prefs):
            return
        
        # 创建UX事件
        ux_event = UXEvent(
            event_type=FeedbackType.PROGRESS,
            message=message,
            tone=user_prefs["tone"],
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            correlation_id=correlation_id,
            data={
                "progress": progress,
                "task_name": task.name,
                "estimated_time": self._estimate_remaining_time(task_id, progress)
            }
        )
        
        # 发送事件
        await self._send_ux_event(ux_event)
        
        # 可能发送鼓励消息
        if progress > 0 and progress % 25 == 0:  # 每25%发送鼓励
            await self._send_encouragement_message(task_id, progress, correlation_id)
    
    async def send_estimate_update(self, task_id: str, estimated_time: float, correlation_id: str = None) -> None:
        """发送估算时间更新"""
        if not self.config.show_estimated_time:
            return
        
        # 生成估算消息
        message = self._generate_estimate_message(estimated_time)
        
        user_prefs = self.get_user_preferences(correlation_id or "")
        
        ux_event = UXEvent(
            event_type=FeedbackType.ESTIMATE,
            message=message,
            tone=MessageTone.NEUTRAL,
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            correlation_id=correlation_id,
            data={"estimated_time": estimated_time}
        )
        
        await self._send_ux_event(ux_event)
    
    async def send_queue_info(self, task_id: str, correlation_id: str = None) -> None:
        """发送队列信息"""
        if not self.config.show_queue_position:
            return
        
        task = self.task_manager.get_task(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return
        
        queue_position = self.task_manager.get_queue_position(task_id)
        if queue_position is None:
            return
        
        # 生成队列消息
        message = self._generate_queue_message(queue_position)
        
        user_prefs = self.get_user_preferences(correlation_id or "")
        
        ux_event = UXEvent(
            event_type=FeedbackType.QUEUE_INFO,
            message=message,
            tone=MessageTone.NEUTRAL,
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            correlation_id=correlation_id,
            data={
                "queue_position": queue_position,
                "estimated_wait_time": self._estimate_wait_time(queue_position)
            }
        )
        
        await self._send_ux_event(ux_event)
    
    async def send_motivation_message(self, task_id: str, correlation_id: str = None) -> None:
        """发送激励消息"""
        if not self.motivation_enabled:
            return
        
        # 检查是否应该发送激励消息
        if not self._should_send_motivation(task_id, correlation_id):
            return
        
        message = random.choice(self.templates.MOTIVATION_MESSAGES)
        user_prefs = self.get_user_preferences(correlation_id or "")
        
        ux_event = UXEvent(
            event_type=FeedbackType.MOTIVATION,
            message=message,
            tone=MessageTone.ENCOURAGING,
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            correlation_id=correlation_id,
            priority=2
        )
        
        await self._send_ux_event(ux_event)
        
        # 记录激励消息发送时间
        key = f"{correlation_id}:{task_id}"
        self.last_motivation_time[key] = datetime.now(timezone.utc)
    
    async def send_warning_message(self, task_id: str, warning: str, correlation_id: str = None) -> None:
        """发送警告消息"""
        user_prefs = self.get_user_preferences(correlation_id or "")
        
        ux_event = UXEvent(
            event_type=FeedbackType.WARNING,
            message=warning,
            tone=MessageTone.URGENT,
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            correlation_id=correlation_id,
            priority=4
        )
        
        await self._send_ux_event(ux_event)
    
    async def send_success_message(self, task_id: str, message: str = None, correlation_id: str = None) -> None:
        """发送成功消息"""
        if not message:
            message = "任务已成功完成！"
        
        user_prefs = self.get_user_preferences(correlation_id or "")
        
        ux_event = UXEvent(
            event_type=FeedbackType.SUCCESS,
            message=message,
            tone=MessageTone.ENCOURAGING,
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            correlation_id=correlation_id,
            priority=3
        )
        
        await self._send_ux_event(ux_event)
    
    def _generate_progress_message(self, progress: float) -> str:
        """生成进度消息"""
        # 找到最接近的进度阈值
        thresholds = [0, 10, 25, 50, 75, 90, 100]
        closest_threshold = min(thresholds, key=lambda x: abs(x - progress))
        
        messages = self.templates.PROGRESS_MESSAGES[closest_threshold]
        message = random.choice(messages)
        
        # 替换进度占位符
        if "{progress}" in message:
            message = message.format(progress=int(progress))
        
        return message
    
    def _generate_estimate_message(self, estimated_time: float) -> str:
        """生成估算消息"""
        template = random.choice(self.templates.ESTIMATE_MESSAGES)
        
        if estimated_time < 60:
            time_str = f"{int(estimated_time)}"
        else:
            minutes = estimated_time / 60
            time_str = f"{minutes:.1f} 分钟"
        
        return template.format(time=time_str)
    
    def _generate_queue_message(self, queue_position: int) -> str:
        """生成队列消息"""
        if queue_position == 0:
            return "即将开始处理您的任务"
        
        template = random.choice(self.templates.QUEUE_MESSAGES)
        
        if "{count}" in template:
            template = template.format(count=queue_position + 1)
        if "{position}" in template:
            template = template.format(position=queue_position + 1)
        if "{time}" in template:
            wait_time = self._estimate_wait_time(queue_position)
            if wait_time < 60:
                time_str = f"{int(wait_time)}"
            else:
                time_str = f"{wait_time/60:.1f} 分钟"
            template = template.format(time=time_str)
        
        return template
    
    def _estimate_remaining_time(self, task_id: str, progress: float) -> Optional[float]:
        """估算剩余时间"""
        task = self.task_manager.get_task(task_id)
        if not task or not task.started_at:
            return None
        
        elapsed = (datetime.now(timezone.utc) - task.started_at).total_seconds()
        
        if progress <= 0:
            return None
        
        # 简单的线性估算
        total_estimated = elapsed * 100 / progress
        remaining = total_estimated - elapsed
        
        return max(remaining, 0)
    
    def _estimate_wait_time(self, queue_position: int) -> float:
        """估算等待时间"""
        # 简单的估算：队列位置 * 平均任务时间
        avg_task_time = 30.0  # 假设平均任务30秒
        return queue_position * avg_task_time
    
    def _should_send_message(self, task_id: str, progress: float, user_prefs: Dict[str, Any]) -> bool:
        """判断是否应该发送消息"""
        frequency = user_prefs.get("message_frequency", "normal")
        
        if frequency == "low":
            return progress in [0, 25, 50, 75, 100]
        elif frequency == "high":
            return progress % 10 == 0 or progress in [0, 100]
        else:  # normal
            return progress in [0, 10, 25, 50, 75, 90, 100]
    
    def _should_send_motivation(self, task_id: str, correlation_id: str) -> bool:
        """判断是否应该发送激励消息"""
        if not correlation_id:
            return False
        
        key = f"{correlation_id}:{task_id}"
        last_time = self.last_motivation_time.get(key)
        
        if not last_time:
            return True
        
        # 至少间隔2分钟才能发送新的激励消息
        return (datetime.now(timezone.utc) - last_time).total_seconds() > 120
    
    async def _send_ux_event(self, ux_event: UXEvent) -> None:
        """发送UX事件"""
        # 添加到历史记录
        if ux_event.correlation_id not in self.message_history:
            self.message_history[ux_event.correlation_id] = []
        
        self.message_history[ux_event.correlation_id].append(ux_event)
        
        # 保持历史记录在合理范围内
        if len(self.message_history[ux_event.correlation_id]) > 100:
            self.message_history[ux_event.correlation_id].pop(0)
        
        # 发送事件
        await event_system.emit_ux_feedback(
            ux_event.event_type.value,
            ux_event.message,
            ux_event.data,
            ux_event.correlation_id
        )
    
    async def _send_encouragement_message(self, task_id: str, progress: float, correlation_id: str = None) -> None:
        """发送鼓励消息"""
        if progress < 25:
            return
        
        message = random.choice(self.templates.ENCOURAGEMENT_MESSAGES)
        
        ux_event = UXEvent(
            event_type=FeedbackType.PROGRESS,
            message=message,
            tone=MessageTone.ENCOURAGING,
            timestamp=datetime.now(timezone.utc),
            task_id=task_id,
            correlation_id=correlation_id,
            data={"progress": progress, "encouragement": True}
        )
        
        await self._send_ux_event(ux_event)
    
    def get_message_history(self, correlation_id: str, limit: int = 50) -> List[UXEvent]:
        """获取消息历史"""
        return self.message_history.get(correlation_id, [])[-limit:]
    
    def get_ux_stats(self) -> Dict[str, Any]:
        """获取UX统计"""
        total_messages = sum(len(history) for history in self.message_history.values())
        
        event_type_counts = {}
        for history in self.message_history.values():
            for event in history:
                event_type = event.event_type.value
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        return {
            "total_messages": total_messages,
            "active_users": len(self.message_history),
            "event_type_distribution": event_type_counts,
            "motivation_enabled": self.motivation_enabled,
            "user_preferences_count": len(self.user_preferences)
        }


class UserExperienceOptimizer:
    """用户体验优化器（主类）"""
    
    def __init__(self, task_manager: TaskManager, config: Optional[NonBlockingConfig] = None):
        self.optimizer = UXOptimizer(task_manager, config)
        self.motivation_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self) -> None:
        """启动用户体验优化器"""
        if self.running:
            return
        
        self.running = True
        self.motivation_task = asyncio.create_task(self._motivation_loop())
        logging.info("UserExperienceOptimizer started")
    
    async def stop(self) -> None:
        """停止用户体验优化器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.motivation_task:
            self.motivation_task.cancel()
            try:
                await self.motivation_task
            except asyncio.CancelledError:
                pass
        
        logging.info("UserExperienceOptimizer stopped")
    
    async def _motivation_loop(self) -> None:
        """激励消息循环"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                # 为长时间运行的任务发送激励消息
                running_tasks = self.task_manager.get_tasks_by_status(TaskStatus.RUNNING)
                for task in running_tasks:
                    if task.duration and task.duration > 60:  # 运行超过1分钟
                        await self.optimizer.send_motivation_message(task.id, task.correlation_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Motivation loop error: {e}")
    
    # 委托方法给optimizer
    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        self.optimizer.set_user_preferences(user_id, preferences)
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        return self.optimizer.get_user_preferences(user_id)
    
    async def send_progress_update(self, task_id: str, progress: float, message: str = None, correlation_id: str = None) -> None:
        await self.optimizer.send_progress_update(task_id, progress, message, correlation_id)
    
    async def send_estimate_update(self, task_id: str, estimated_time: float, correlation_id: str = None) -> None:
        await self.optimizer.send_estimate_update(task_id, estimated_time, correlation_id)
    
    async def send_queue_info(self, task_id: str, correlation_id: str = None) -> None:
        await self.optimizer.send_queue_info(task_id, correlation_id)
    
    async def send_warning_message(self, task_id: str, warning: str, correlation_id: str = None) -> None:
        await self.optimizer.send_warning_message(task_id, warning, correlation_id)
    
    async def send_success_message(self, task_id: str, message: str = None, correlation_id: str = None) -> None:
        await self.optimizer.send_success_message(task_id, message, correlation_id)
    
    def get_message_history(self, correlation_id: str, limit: int = 50) -> List[UXEvent]:
        return self.optimizer.get_message_history(correlation_id, limit)
    
    def get_stats(self) -> Dict[str, Any]:
        return self.optimizer.get_ux_stats()