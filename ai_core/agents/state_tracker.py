"""
状态跟踪器

实现任务状态跟踪和进度监控功能
"""

import json
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import uuid

from .models import Task, TaskStatus, TaskExecutionPlan


@dataclass
class TaskEvent:
    """任务事件"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    event_type: str = ""  # created, started, progress, completed, failed, cancelled
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressSnapshot:
    """进度快照"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0
    overall_progress: float = 0.0
    estimated_completion: Optional[datetime] = None
    task_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class StateTracker:
    """任务状态跟踪器"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        
        # 任务状态存储
        self.tasks: Dict[str, Task] = {}
        self.task_events: Dict[str, List[TaskEvent]] = defaultdict(list)
        self.execution_plans: Dict[str, TaskExecutionPlan] = {}
        
        # 进度跟踪
        self.progress_history: deque = deque(maxlen=max_history_size)
        self.current_snapshot: Optional[ProgressSnapshot] = None
        
        # 监控配置
        self.monitoring_enabled = True
        self.update_interval = 1.0  # 秒
        self.monitor_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        
        # 回调函数
        self.on_task_status_change: Optional[Callable] = None
        self.on_progress_update: Optional[Callable] = None
        self.on_milestone_reached: Optional[Callable] = None
        
        # 统计信息
        self.start_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        self.total_events_processed = 0
        
        # 性能指标
        self.performance_metrics: Dict[str, Any] = {
            "average_task_duration": 0.0,
            "success_rate": 0.0,
            "throughput": 0.0,  # tasks per hour
            "bottleneck_tasks": [],
            "resource_utilization": {}
        }
    
    def register_task(self, task: Task) -> str:
        """注册任务"""
        self.tasks[task.id] = task
        
        # 记录任务创建事件
        self._record_event(TaskEvent(
            task_id=task.id,
            event_type="created",
            data={
                "name": task.name,
                "task_type": task.task_type,
                "priority": task.priority.value,
                "estimated_duration": task.estimated_duration
            }
        ))
        
        # 更新进度快照
        self._update_progress_snapshot()
        
        return task.id
    
    def update_task_status(
        self,
        task_id: str,
        new_status: TaskStatus,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新任务状态"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        old_status = task.status
        
        # 更新任务状态
        task.update_status(new_status, error_message)
        
        if progress is not None:
            task.update_progress(progress)
        
        if additional_data:
            task.metadata.update(additional_data)
        
        # 记录状态变更事件
        event_data = {
            "old_status": old_status.value,
            "new_status": new_status.value,
            "progress": task.progress
        }
        
        if error_message:
            event_data["error_message"] = error_message
        
        if additional_data:
            event_data.update(additional_data)
        
        self._record_event(TaskEvent(
            task_id=task_id,
            event_type="status_changed",
            data=event_data
        ))
        
        # 触发状态变更回调
        if self.on_task_status_change:
            self._safe_callback(self.on_task_status_change, task, old_status, new_status)
        
        # 检查里程碑
        self._check_milestones()
        
        # 更新进度快照
        self._update_progress_snapshot()
        
        return True
    
    def update_task_progress(self, task_id: str, progress: float, additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """更新任务进度"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        old_progress = task.progress
        
        task.update_progress(progress)
        
        if additional_data:
            task.metadata.update(additional_data)
        
        # 记录进度更新事件
        event_data = {
            "old_progress": old_progress,
            "new_progress": progress,
            "progress_change": progress - old_progress
        }
        
        if additional_data:
            event_data.update(additional_data)
        
        self._record_event(TaskEvent(
            task_id=task_id,
            event_type="progress_updated",
            data=event_data
        ))
        
        # 触发进度更新回调
        if self.on_progress_update:
            self._safe_callback(self.on_progress_update, task, old_progress, progress)
        
        # 更新进度快照
        self._update_progress_snapshot()
        
        return True
    
    def register_execution_plan(self, plan: TaskExecutionPlan) -> str:
        """注册执行计划"""
        self.execution_plans[plan.id] = plan
        
        # 注册计划中的所有任务
        for task in plan.tasks:
            self.register_task(task)
        
        # 记录计划创建事件
        self._record_event(TaskEvent(
            task_id=plan.id,
            event_type="plan_created",
            data={
                "plan_name": plan.name,
                "total_tasks": len(plan.tasks),
                "execution_mode": plan.execution_mode
            }
        ))
        
        return plan.id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            "task": task.to_dict(),
            "events": [event.__dict__ for event in self.task_events[task_id]],
            "current_snapshot": self.current_snapshot.__dict__ if self.current_snapshot else None
        }
    
    def get_execution_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """获取执行计划状态"""
        if plan_id not in self.execution_plans:
            return None
        
        plan = self.execution_plans[plan_id]
        
        # 统计任务状态
        status_counts = defaultdict(int)
        for task in plan.tasks:
            status_counts[task.status.value] += 1
        
        return {
            "plan": {
                "id": plan.id,
                "name": plan.name,
                "status": plan.status.value,
                "progress": plan.progress,
                "total_tasks": len(plan.tasks),
                "created_at": plan.created_at.isoformat()
            },
            "task_status_distribution": dict(status_counts),
            "current_snapshot": self.current_snapshot.__dict__ if self.current_snapshot else None,
            "performance_metrics": self.performance_metrics
        }
    
    def get_progress_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取进度历史"""
        history = list(self.progress_history)[-limit:]
        return [snapshot.__dict__ for snapshot in history]
    
    def get_task_events(
        self,
        task_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取任务事件"""
        events = []
        
        if task_id:
            events.extend(self.task_events[task_id])
        else:
            for task_events in self.task_events.values():
                events.extend(task_events)
        
        # 过滤事件
        filtered_events = []
        for event in events:
            if event_type and event.event_type != event_type:
                continue
            
            if start_time and event.timestamp < start_time:
                continue
            
            if end_time and event.timestamp > end_time:
                continue
            
            filtered_events.append(event)
        
        # 按时间排序并限制数量
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return [event.__dict__ for event in filtered_events[:limit]]
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.start_time = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._update_progress_snapshot()
                self._calculate_performance_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"监控循环异常: {e}")
                time.sleep(1)
    
    def _update_progress_snapshot(self):
        """更新进度快照"""
        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return
        
        # 统计任务状态
        status_counts = defaultdict(int)
        total_progress = 0.0
        
        for task in self.tasks.values():
            status_counts[task.status] += 1
            total_progress += task.progress
        
        # 计算预计完成时间
        estimated_completion = None
        if status_counts[TaskStatus.RUNNING] > 0:
            # 基于当前进度计算预计完成时间
            overall_progress = total_progress / total_tasks
            if overall_progress > 0 and self.start_time:
                elapsed_time = datetime.now() - self.start_time
                total_estimated_time = elapsed_time / overall_progress
                estimated_completion = self.start_time + total_estimated_time
        
        # 创建快照
        snapshot = ProgressSnapshot(
            total_tasks=total_tasks,
            completed_tasks=status_counts[TaskStatus.COMPLETED],
            failed_tasks=status_counts[TaskStatus.FAILED],
            running_tasks=status_counts[TaskStatus.RUNNING],
            pending_tasks=status_counts[TaskStatus.PENDING],
            overall_progress=total_progress / total_tasks,
            estimated_completion=estimated_completion,
            task_details={
                task_id: {
                    "status": task.status.value,
                    "progress": task.progress,
                    "priority": task.priority.value,
                    "elapsed_time": (datetime.now() - task.created_at).total_seconds()
                }
                for task_id, task in self.tasks.items()
            }
        )
        
        self.current_snapshot = snapshot
        self.progress_history.append(snapshot)
        self.last_update_time = datetime.now()
    
    def _calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.tasks:
            return
        
        # 计算平均任务时长
        completed_tasks = [task for task in self.tasks.values() if task.actual_duration]
        if completed_tasks:
            self.performance_metrics["average_task_duration"] = sum(
                task.actual_duration for task in completed_tasks
            ) / len(completed_tasks)
        
        # 计算成功率
        total_finished = len([task for task in self.tasks.values() 
                             if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]])
        if total_finished > 0:
            successful = len([task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED])
            self.performance_metrics["success_rate"] = successful / total_finished
        
        # 计算吞吐量
        if self.start_time:
            elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if elapsed_hours > 0:
                self.performance_metrics["throughput"] = len(completed_tasks) / elapsed_hours
        
        # 识别瓶颈任务
        running_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.RUNNING]
        self.performance_metrics["bottleneck_tasks"] = [
            task.id for task in running_tasks
            if task.estimated_duration and task.actual_duration and 
            task.actual_duration > task.estimated_duration * 1.5
        ]
    
    def _check_milestones(self):
        """检查里程碑"""
        if not self.current_snapshot:
            return
        
        # 检查完成率里程碑
        completion_rate = self.current_snapshot.overall_progress
        
        milestones = [
            (0.25, "25% 完成"),
            (0.50, "50% 完成"),
            (0.75, "75% 完成"),
            (1.00, "100% 完成")
        ]
        
        for threshold, description in milestones:
            if completion_rate >= threshold:
                self._record_event(TaskEvent(
                    task_id="system",
                    event_type="milestone_reached",
                    data={
                        "milestone": description,
                        "completion_rate": completion_rate,
                        "timestamp": datetime.now().isoformat()
                    }
                ))
                
                if self.on_milestone_reached:
                    self._safe_callback(self.on_milestone_reached, description, completion_rate)
    
    def _record_event(self, event: TaskEvent):
        """记录事件"""
        self.task_events[event.task_id].append(event)
        self.total_events_processed += 1
        
        # 限制事件历史大小
        if len(self.task_events[event.task_id]) > self.max_history_size:
            self.task_events[event.task_id] = self.task_events[event.task_id][-self.max_history_size:]
    
    def _safe_callback(self, callback: Callable, *args, **kwargs):
        """安全执行回调函数"""
        try:
            if asyncio.iscoroutinefunction(callback):
                # 如果是异步函数，创建新的事件循环执行
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(callback(*args, **kwargs))
                finally:
                    loop.close()
            else:
                # 同步函数直接执行
                callback(*args, **kwargs)
        except Exception as e:
            print(f"回调函数执行失败: {e}")
    
    def export_status_report(self, format_type: str = "json") -> str:
        """导出状态报告"""
        if format_type == "json":
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tasks": len(self.tasks),
                    "current_snapshot": self.current_snapshot.__dict__ if self.current_snapshot else None,
                    "performance_metrics": self.performance_metrics,
                    "monitoring_duration": (
                        datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                },
                "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
                "recent_events": self.get_task_events(limit=50)
            }
            return json.dumps(report, ensure_ascii=False, indent=2, default=str)
        else:
            # 可以扩展其他格式
            return self.export_status_report("json")
    
    def reset(self):
        """重置状态跟踪器"""
        self.tasks.clear()
        self.task_events.clear()
        self.execution_plans.clear()
        self.progress_history.clear()
        self.current_snapshot = None
        self.start_time = None
        self.last_update_time = None
        self.total_events_processed = 0
        self.performance_metrics = {
            "average_task_duration": 0.0,
            "success_rate": 0.0,
            "throughput": 0.0,
            "bottleneck_tasks": [],
            "resource_utilization": {}
        }