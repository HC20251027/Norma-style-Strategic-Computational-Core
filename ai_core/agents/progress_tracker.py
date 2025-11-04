"""
进度跟踪器
实现任务进度的实时跟踪和在对话中的显示功能
"""

import asyncio
import uuid
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import time


class ProgressStatus(Enum):
    """进度状态"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressStage(Enum):
    """进度阶段"""
    INITIALIZATION = "initialization"
    PROCESSING = "processing"
    VALIDATION = "validation"
    FINALIZATION = "finalization"
    COMPLETION = "completion"


class ProgressUpdate:
    """进度更新数据类"""
    
    def __init__(self, 
                 task_id: str,
                 session_id: str,
                 progress: float,
                 status: ProgressStatus,
                 message: str = "",
                 stage: Optional[ProgressStage] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.task_id = task_id
        self.session_id = session_id
        self.progress = max(0, min(100, progress))  # 限制在0-100之间
        self.status = status
        self.message = message
        self.stage = stage
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.estimated_completion: Optional[datetime] = None
        self.actual_duration: Optional[float] = None


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, update_interval: float = 0.5):
        self.update_interval = update_interval
        self.active_tracking: Dict[str, Dict[str, Any]] = {}
        self.progress_history: Dict[str, List[ProgressUpdate]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        
        # 进度统计
        self.stats = {
            'total_tracked_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_completion_time': 0,
            'total_updates_sent': 0
        }
        
        # 进度更新循环
        self.update_loop_running = False
        self.update_loop_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def start_tracking(self, 
                           task_id: str, 
                           session_id: str,
                           initial_progress: float = 0,
                           estimated_duration: Optional[int] = None) -> bool:
        """开始跟踪任务进度"""
        if task_id in self.active_tracking:
            return False
        
        tracking_data = {
            'task_id': task_id,
            'session_id': session_id,
            'start_time': datetime.now(),
            'initial_progress': initial_progress,
            'current_progress': initial_progress,
            'estimated_duration': estimated_duration,
            'status': ProgressStatus.IN_PROGRESS,
            'stage': ProgressStage.INITIALIZATION,
            'update_count': 0,
            'last_update': datetime.now(),
            'milestones': [],
            'custom_data': {}
        }
        
        self.active_tracking[task_id] = tracking_data
        self.progress_history[task_id] = []
        
        # 创建初始进度更新
        initial_update = ProgressUpdate(
            task_id=task_id,
            session_id=session_id,
            progress=initial_progress,
            status=ProgressStatus.IN_PROGRESS,
            message="任务开始执行",
            stage=ProgressStage.INITIALIZATION
        )
        
        await self._record_progress_update(initial_update)
        
        # 启动更新循环
        if not self.update_loop_running:
            await self._start_update_loop()
        
        self.stats['total_tracked_tasks'] += 1
        self.logger.info(f"Started tracking progress for task {task_id}")
        return True
    
    async def update_progress(self, 
                            task_id: str, 
                            progress: float,
                            message: str = "",
                            stage: Optional[ProgressStage] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新任务进度"""
        if task_id not in self.active_tracking:
            return False
        
        tracking_data = self.active_tracking[task_id]
        
        # 更新进度数据
        tracking_data['current_progress'] = progress
        tracking_data['last_update'] = datetime.now()
        tracking_data['update_count'] += 1
        
        if stage:
            tracking_data['stage'] = stage
        
        if message:
            tracking_data['last_message'] = message
        
        if metadata:
            tracking_data['custom_data'].update(metadata)
        
        # 计算预计完成时间
        if tracking_data['estimated_duration']:
            elapsed = (datetime.now() - tracking_data['start_time']).total_seconds()
            if progress > 0:
                estimated_total = elapsed / (progress / 100)
                remaining = estimated_total - elapsed
                estimated_completion = datetime.now() + timedelta(seconds=remaining)
                tracking_data['estimated_completion'] = estimated_completion
        
        # 创建进度更新
        status = self._determine_status(progress, tracking_data)
        update = ProgressUpdate(
            task_id=task_id,
            session_id=tracking_data['session_id'],
            progress=progress,
            status=status,
            message=message,
            stage=stage,
            metadata=metadata
        )
        
        if tracking_data.get('estimated_completion'):
            update.estimated_completion = tracking_data['estimated_completion']
        
        await self._record_progress_update(update)
        
        # 检查里程碑
        await self._check_milestones(task_id, progress)
        
        self.logger.debug(f"Updated progress for task {task_id}: {progress}%")
        return True
    
    async def complete_task(self, task_id: str, final_message: str = "任务已完成") -> bool:
        """完成任务"""
        if task_id not in self.active_tracking:
            return False
        
        tracking_data = self.active_tracking[task_id]
        tracking_data['current_progress'] = 100
        tracking_data['status'] = ProgressStatus.COMPLETED
        tracking_data['stage'] = ProgressStage.COMPLETION
        tracking_data['completion_time'] = datetime.now()
        
        # 计算实际耗时
        elapsed = (tracking_data['completion_time'] - tracking_data['start_time']).total_seconds()
        tracking_data['actual_duration'] = elapsed
        
        # 创建完成更新
        completion_update = ProgressUpdate(
            task_id=task_id,
            session_id=tracking_data['session_id'],
            progress=100,
            status=ProgressStatus.COMPLETED,
            message=final_message,
            stage=ProgressStage.COMPLETION
        )
        completion_update.actual_duration = elapsed
        
        await self._record_progress_update(completion_update)
        
        # 更新统计
        self.stats['completed_tasks'] += 1
        self._update_average_completion_time(elapsed)
        
        self.logger.info(f"Completed tracking for task {task_id}")
        return True
    
    async def fail_task(self, task_id: str, error_message: str = "任务执行失败") -> bool:
        """标记任务失败"""
        if task_id not in self.active_tracking:
            return False
        
        tracking_data = self.active_tracking[task_id]
        tracking_data['status'] = ProgressStatus.FAILED
        tracking_data['failure_time'] = datetime.now()
        tracking_data['error_message'] = error_message
        
        # 创建失败更新
        failure_update = ProgressUpdate(
            task_id=task_id,
            session_id=tracking_data['session_id'],
            progress=tracking_data['current_progress'],
            status=ProgressStatus.FAILED,
            message=error_message,
            metadata={'error': True}
        )
        
        await self._record_progress_update(failure_update)
        
        # 更新统计
        self.stats['failed_tasks'] += 1
        
        self.logger.error(f"Failed tracking for task {task_id}: {error_message}")
        return True
    
    async def pause_tracking(self, task_id: str) -> bool:
        """暂停进度跟踪"""
        if task_id not in self.active_tracking:
            return False
        
        tracking_data = self.active_tracking[task_id]
        tracking_data['status'] = ProgressStatus.PAUSED
        tracking_data['pause_time'] = datetime.now()
        
        pause_update = ProgressUpdate(
            task_id=task_id,
            session_id=tracking_data['session_id'],
            progress=tracking_data['current_progress'],
            status=ProgressStatus.PAUSED,
            message="任务已暂停"
        )
        
        await self._record_progress_update(pause_update)
        return True
    
    async def resume_tracking(self, task_id: str) -> bool:
        """恢复进度跟踪"""
        if task_id not in self.active_tracking:
            return False
        
        tracking_data = self.active_tracking[task_id]
        tracking_data['status'] = ProgressStatus.IN_PROGRESS
        tracking_data['resume_time'] = datetime.now()
        
        resume_update = ProgressUpdate(
            task_id=task_id,
            session_id=tracking_data['session_id'],
            progress=tracking_data['current_progress'],
            status=ProgressStatus.IN_PROGRESS,
            message="任务已恢复"
        )
        
        await self._record_progress_update(resume_update)
        return True
    
    async def stop_tracking(self, task_id: str) -> bool:
        """停止进度跟踪"""
        if task_id not in self.active_tracking:
            return False
        
        tracking_data = self.active_tracking[task_id]
        
        # 如果任务未完成，标记为已取消
        if tracking_data['status'] == ProgressStatus.IN_PROGRESS:
            await self.fail_task(task_id, "任务被取消")
        
        # 移动到历史记录
        self.progress_history[task_id] = self.progress_history.get(task_id, [])
        
        # 清理活跃跟踪
        del self.active_tracking[task_id]
        
        self.logger.info(f"Stopped tracking for task {task_id}")
        return True
    
    async def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务进度"""
        if task_id not in self.active_tracking:
            # 检查历史记录
            if task_id in self.progress_history and self.progress_history[task_id]:
                latest_update = self.progress_history[task_id][-1]
                return {
                    'task_id': task_id,
                    'progress': latest_update.progress,
                    'status': latest_update.status.value,
                    'message': latest_update.message,
                    'stage': latest_update.stage.value if latest_update.stage else None,
                    'timestamp': latest_update.timestamp.isoformat(),
                    'completed': True
                }
            return None
        
        tracking_data = self.active_tracking[task_id]
        
        # 计算剩余时间
        remaining_time = None
        if tracking_data.get('estimated_completion'):
            remaining_time = (tracking_data['estimated_completion'] - datetime.now()).total_seconds()
            remaining_time = max(0, remaining_time)
        
        return {
            'task_id': task_id,
            'session_id': tracking_data['session_id'],
            'progress': tracking_data['current_progress'],
            'status': tracking_data['status'].value,
            'stage': tracking_data['stage'].value if tracking_data['stage'] else None,
            'message': tracking_data.get('last_message', ''),
            'start_time': tracking_data['start_time'].isoformat(),
            'last_update': tracking_data['last_update'].isoformat(),
            'estimated_completion': tracking_data.get('estimated_completion', {}).isoformat() if tracking_data.get('estimated_completion') else None,
            'remaining_time': remaining_time,
            'update_count': tracking_data['update_count'],
            'milestones': tracking_data['milestones'],
            'custom_data': tracking_data['custom_data'],
            'completed': tracking_data['status'] in [ProgressStatus.COMPLETED, ProgressStatus.FAILED]
        }
    
    async def get_session_progress(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话的所有任务进度"""
        session_progress = []
        
        for task_id, tracking_data in self.active_tracking.items():
            if tracking_data['session_id'] == session_id:
                progress = await self.get_progress(task_id)
                if progress:
                    session_progress.append(progress)
        
        return session_progress
    
    async def get_progress_history(self, task_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取任务进度历史"""
        if task_id not in self.progress_history:
            return []
        
        history = self.progress_history[task_id]
        
        # 转换为字典格式
        history_dicts = []
        for update in history:
            history_dicts.append({
                'id': update.id,
                'task_id': update.task_id,
                'progress': update.progress,
                'status': update.status.value,
                'message': update.message,
                'stage': update.stage.value if update.stage else None,
                'timestamp': update.timestamp.isoformat(),
                'metadata': update.metadata
            })
        
        if limit:
            return history_dicts[-limit:]
        
        return history_dicts
    
    async def add_milestone(self, task_id: str, progress: float, description: str) -> bool:
        """添加里程碑"""
        if task_id not in self.active_tracking:
            return False
        
        milestone = {
            'progress': progress,
            'description': description,
            'timestamp': datetime.now()
        }
        
        self.active_tracking[task_id]['milestones'].append(milestone)
        return True
    
    async def _check_milestones(self, task_id: str, current_progress: float):
        """检查里程碑"""
        tracking_data = self.active_tracking[task_id]
        
        for milestone in tracking_data['milestones']:
            if not milestone.get('reached', False) and current_progress >= milestone['progress']:
                milestone['reached'] = True
                milestone['reached_at'] = datetime.now()
                
                # 发送里程碑通知
                await self._notify_subscribers(task_id, {
                    'type': 'milestone_reached',
                    'milestone': milestone
                })
    
    async def _record_progress_update(self, update: ProgressUpdate):
        """记录进度更新"""
        task_id = update.task_id
        
        # 添加到历史记录
        if task_id not in self.progress_history:
            self.progress_history[task_id] = []
        
        self.progress_history[task_id].append(update)
        
        # 限制历史记录大小
        if len(self.progress_history[task_id]) > 100:
            self.progress_history[task_id] = self.progress_history[task_id][-50:]
        
        # 通知订阅者
        await self._notify_subscribers(task_id, {
            'type': 'progress_update',
            'update': {
                'task_id': update.task_id,
                'progress': update.progress,
                'status': update.status.value,
                'message': update.message,
                'timestamp': update.timestamp.isoformat()
            }
        })
        
        self.stats['total_updates_sent'] += 1
    
    def _determine_status(self, progress: float, tracking_data: Dict[str, Any]) -> ProgressStatus:
        """根据进度确定状态"""
        if progress >= 100:
            return ProgressStatus.COMPLETED
        elif tracking_data['status'] == ProgressStatus.PAUSED:
            return ProgressStatus.PAUSED
        else:
            return ProgressStatus.IN_PROGRESS
    
    def _update_average_completion_time(self, duration: float):
        """更新平均完成时间"""
        if self.stats['completed_tasks'] == 0:
            self.stats['average_completion_time'] = duration
        else:
            total_time = self.stats['average_completion_time'] * (self.stats['completed_tasks'] - 1)
            self.stats['average_completion_time'] = (total_time + duration) / self.stats['completed_tasks']
    
    async def _start_update_loop(self):
        """启动进度更新循环"""
        if self.update_loop_running:
            return
        
        self.update_loop_running = True
        self.update_loop_task = asyncio.create_task(self._update_loop())
    
    async def _stop_update_loop(self):
        """停止进度更新循环"""
        self.update_loop_running = False
        if self.update_loop_task:
            self.update_loop_task.cancel()
            try:
                await self.update_loop_task
            except asyncio.CancelledError:
                pass
    
    async def _update_loop(self):
        """进度更新循环"""
        while self.update_loop_running:
            try:
                current_time = datetime.now()
                
                # 检查超时任务
                timeout_tasks = []
                for task_id, tracking_data in self.active_tracking.items():
                    time_since_update = (current_time - tracking_data['last_update']).total_seconds()
                    if time_since_update > 300:  # 5分钟无更新
                        timeout_tasks.append(task_id)
                
                # 标记超时任务为失败
                for task_id in timeout_tasks:
                    await self.fail_task(task_id, "任务执行超时")
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in progress update loop: {e}")
                await asyncio.sleep(1)
    
    async def subscribe_to_progress(self, task_id: str, callback: Callable):
        """订阅进度更新"""
        if task_id not in self.subscribers:
            self.subscribers[task_id] = []
        self.subscribers[task_id].append(callback)
    
    async def unsubscribe_from_progress(self, task_id: str, callback: Callable):
        """取消订阅进度更新"""
        if task_id in self.subscribers:
            try:
                self.subscribers[task_id].remove(callback)
            except ValueError:
                pass
    
    async def _notify_subscribers(self, task_id: str, data: Dict[str, Any]):
        """通知订阅者"""
        if task_id in self.subscribers:
            for callback in self.subscribers[task_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Error notifying subscriber for task {task_id}: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'active_tracking_count': len(self.active_tracking),
            'total_history_entries': sum(len(history) for history in self.progress_history.values()),
            'subscriber_count': sum(len(subs) for subs in self.subscribers.values())
        }